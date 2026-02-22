from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import json
import time
import asyncio
import sys
import os
import io
from sklearn.ensemble import IsolationForest
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pydantic import BaseModel

# Add parent dir to path so we can import trainer.py which contains `preprocess_spectra`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from trainer import preprocess_spectra
except ImportError:
    # If the file runs locally inside the hackathon folder root.
    sys.path.append(os.getcwd())
    from trainer import preprocess_spectra

app = FastAPI(title="Sugarcane NIR Real-Time Prediction API")

# Enable CORS for the frontend React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
print("Loading Models via Joblib...")
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    pls_model = joblib.load(os.path.join(root_dir, 'pls_model.pkl'))
    wavelengths = joblib.load(os.path.join(root_dir, 'wavelengths.pkl'))
    X_test = np.load(os.path.join(root_dir, 'X_test_raw.npy'))
    y_test = np.load(os.path.join(root_dir, 'y_test.npy'))
    
    # Store clean wavelength float values
    wavelengths_plot = [float(str(w).replace('amplitude-', '')) for w in wavelengths]
    print(f"Loaded successfully. {len(X_test)} samples available for simulation.")
    
    print("Training Anomaly Detector (Isolation Forest) on baseline data...")
    X_test_prep = preprocess_spectra(X_test)
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    iso_forest.fit(X_test_prep)
    print("Anomaly Detector Ready.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please make sure you have run `python trainer.py` first.")
    
def add_industrial_noise(spectrum, noise_level=0.02):
    noise = np.random.normal(0, noise_level, spectrum.shape)
    baseline_drift = np.random.uniform(-0.01, 0.01)
    return spectrum * (1 + noise) + baseline_drift

# Websocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
def read_root():
    return {"status": "Sugarcane NIR Backend API is running."}

@app.get("/api/config")
def get_config():
    """Returns static data required by the frontend initialized state."""
    return {"wavelengths": wavelengths_plot}

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    idx = 0
    try:
        while True:
            # We wait for the client to ask for the next tick, or just stream continuously if client indicates.
            # Here we listen for optional settings updates (like noise level from frontend UI).
            data = await websocket.receive_text()
            config = json.loads(data)
            
            noise_pct = config.get("noiseLevel", 2.0)
            
            # Grab sample
            raw_spectrum = X_test[idx]
            actual_pol = y_test[idx]
            
            # Transform
            noisy_spectrum = add_industrial_noise(raw_spectrum, noise_level=noise_pct/100.0)
            X_live_prep = preprocess_spectra(noisy_spectrum.reshape(1, -1))
            
            start_time = time.time()
            pred_pol = float(np.squeeze(pls_model.predict(X_live_prep)))
            is_anomaly = iso_forest.predict(X_live_prep)[0] == -1
            inference_ms = (time.time() - start_time) * 1000
            
            payload = {
                "timestamp": time.time(),
                "actual_pol": float(actual_pol),
                "predicted_pol": pred_pol,
                "inference_ms": inference_ms,
                "noisy_spectrum": noisy_spectrum.tolist(),
                "alert": pred_pol < config.get("threshold", 13.0),
                "anomaly": bool(is_anomaly)
            }
            
            await websocket.send_text(json.dumps(payload))
            
            # Loop sample index
            idx = (idx + 1) % len(X_test)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected.")

class ReportData(BaseModel):
    timestamp: list[float]
    predicted_pol: list[float]
    actual_pol: list[float]

@app.post("/api/report")
async def generate_report(data: ReportData):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 24)
    c.setFillColorRGB(0.04, 0.72, 0.31) # Green
    c.drawString(50, 750, "Sugarcane NIR Shift Report")
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 700, f"Total Samples Processed: {len(data.predicted_pol)}")
    
    avg_pol = sum(data.predicted_pol) / len(data.predicted_pol) if data.predicted_pol else 0
    c.drawString(50, 670, f"Average Predicted Pol (TS%): {avg_pol:.2f}%")
    
    # Calculate breaches
    threshold = 13.0
    breaches = sum(1 for p in data.predicted_pol if p < threshold)
    c.drawString(50, 640, f"Total Low-Pol Alerts: {breaches}")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 580, "Technical Details:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 560, "- Model: Partial Least Squares (PLS) Regression")
    c.drawString(50, 540, "- Preprocessing: Savitzky-Golay (drv=1) + Standard Normal Variate (SNV)")
    c.drawString(50, 520, "- Deployment: Automated via Edge Device / Web SCADA Mock")
    
    c.save()
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=shift_report.pdf"})
