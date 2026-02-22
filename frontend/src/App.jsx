import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Play, Square, Activity, AlertTriangle, CheckCircle2, TrendingUp, Download, Settings } from 'lucide-react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder } from '@react-three/drei';
import './index.css';

// 3D Digital Twin Component
function SugarcaneBelt({ activeAlert, isRunning }) {
  const meshRef = useRef();

  useFrame((state, delta) => {
    if (meshRef.current && isRunning) {
      meshRef.current.position.x += delta * 2.5; // Belt speed
      if (meshRef.current.position.x > 3) meshRef.current.position.x = -3;
    }
  });

  const caneColor = activeAlert ? "#ff3366" : "#09b850";

  return (
    <group>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 10, 5]} intensity={1} />

      {/* Conveyor Belt */}
      <Box args={[8, 0.2, 2]} position={[0, -0.6, 0]}>
        <meshStandardMaterial color="#21262d" />
      </Box>
      <Box args={[8, 0.8, 1.8]} position={[0, -1.1, 0]}>
        <meshStandardMaterial color="#161b22" />
      </Box>

      {/* Sugarcane Material Flow */}
      <Cylinder ref={meshRef} args={[0.3, 0.3, 1.5]} rotation={[0, 0, Math.PI / 2]} position={[-3, -0.2, 0]}>
        <meshStandardMaterial color={caneColor} emissive={caneColor} emissiveIntensity={0.4} />
      </Cylinder>

      {/* NIR Sensor Block Array */}
      <Box args={[0.5, 0.5, 2.5]} position={[0, 1.5, 0]}>
        <meshStandardMaterial color="#30363d" />
      </Box>

      {/* NIR Laser Sensor Field */}
      {isRunning && (
        <Box args={[0.05, 3, 2]} position={[0, 0, 0]}>
          <meshStandardMaterial color="#00d2ff" emissive="#00d2ff" emissiveIntensity={isRunning ? 2 : 0} transparent opacity={0.3} />
        </Box>
      )}

      {/* Set Default Angle */}
      <OrbitControls enableZoom={false} autoRotate={isRunning} autoRotateSpeed={-0.5} maxPolarAngle={Math.PI / 2.5} minPolarAngle={Math.PI / 3} />
    </group>
  );
}

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [noiseLevel, setNoiseLevel] = useState(2.0);
  const [threshold, setThreshold] = useState(13.0);

  const [wavelengths, setWavelengths] = useState([]);
  const [currentSpectrum, setCurrentSpectrum] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [latestData, setLatestData] = useState(null);
  const [logs, setLogs] = useState("");

  const ws = useRef(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/config')
      .then(res => res.json())
      .then(data => setWavelengths(data.wavelengths))
      .catch(err => console.error("Config fetch error:", err));
  }, []);

  useEffect(() => {
    if (isRunning) {
      ws.current = new WebSocket('ws://localhost:8000/ws/simulation');

      ws.current.onopen = () => sendConfig();

      ws.current.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        handlePayload(payload);

        if (ws.current?.readyState === WebSocket.OPEN && isRunning) {
          setTimeout(sendConfig, 1000);
        }
      };

      return () => ws.current?.close();
    }
  }, [isRunning]);

  const sendConfig = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ noiseLevel, threshold }));
    }
  }

  const handlePayload = (data) => {
    setLatestData(data);

    const spectrumData = wavelengths.map((wv, i) => ({
      wavelength: wv,
      absorbance: data.noisy_spectrum[0][i] || data.noisy_spectrum[i]
    }));
    setCurrentSpectrum(spectrumData);

    setPredictions(prev => {
      const updated = [...prev, {
        time: prev.length,
        predicted: data.predicted_pol,
        actual: data.actual_pol
      }];
      if (updated.length > 20) updated.shift();
      return updated.map((item, idx) => ({ ...item, time: idx }));
    });

    const mqttMsg = {
      id: "NIR_01",
      pol: Number(data.predicted_pol.toFixed(2)),
      status: data.anomaly ? "ERR_ANOMALY" : (data.alert ? "ALERT" : "OK")
    };
    setLogs(prev => {
      const newLog = `PUB: topic/sugar/mill1\n${JSON.stringify(mqttMsg, null, 2)}\n\n` + prev;
      return newLog.substring(0, 800);
    });
  };

  const downloadReport = async () => {
    if (predictions.length === 0) {
      alert("No data collected yet! Run simulation first.");
      return;
    }
    const data = {
      timestamp: predictions.map(p => p.time),
      actual_pol: predictions.map(p => p.actual),
      predicted_pol: predictions.map(p => p.predicted)
    };

    try {
      const response = await fetch("http://localhost:8000/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "shift_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error("Report Generation failed", e);
    }
  };

  return (
    <div className="dashboard-container">
      <h1 className="header-title">Sugarcane NIR Real-Time Platform</h1>
      <div className="header-subtitle">
        Automated Chemometrics, Predictive Maintenance & Digital Twin Demo
      </div>

      <div className="top-grid" style={{ gridTemplateColumns: '1fr 1fr 2fr' }}>
        {/* Controls Card */}
        <div className="card">
          <h3><Settings size={20} /> Validation Controls</h3>

          <div className="control-group">
            <label>Industrial Noise Simulation: {noiseLevel}%</label>
            <input
              type="range" min="0" max="10" step="0.5"
              value={noiseLevel} onChange={(e) => setNoiseLevel(Number(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label>Pol Alert Threshold: {threshold}%</label>
            <input
              type="range" min="10" max="25" step="0.5"
              value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>

          <div style={{ display: 'flex', gap: '10px', marginTop: '1rem' }}>
            <button
              className="btn-primary"
              onClick={() => setIsRunning(!isRunning)}
              style={{ background: isRunning ? 'linear-gradient(45deg, #ff3366, #ff7b00)' : '', flex: 1 }}
            >
              {isRunning ? <Square size={18} /> : <Play size={18} />}
            </button>

            <button
              className="btn-primary"
              onClick={downloadReport}
              style={{ background: 'linear-gradient(45deg, #4b5263, #30363d)', flex: 1 }}
              title="Generate Plant Manager PDF Shift Report"
            >
              <Download size={18} /> PDF
            </button>
          </div>

          <div style={{ marginTop: '1.5rem' }}>
            <label style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem', display: 'block' }}>MQTT Output Log</label>
            <div className="console-mock">
              {logs || "Awaiting simulation..."}
            </div>
          </div>
        </div>

        {/* Metric and Anomaly Card */}
        <div className="card">
          <h3><TrendingUp size={20} /> Live Pol & Telemetry</h3>
          <div className="metric-wrapper">
            <span style={{ color: 'var(--text-secondary)' }}>Predicted TS / Pol %</span>

            <div className={`metric-value ${latestData?.alert ? 'low-pol' : 'high-pol'}`}>
              {latestData ? latestData.predicted_pol.toFixed(2) : "--.--"} %
            </div>

            {latestData && (
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Lab Ref: {latestData.actual_pol.toFixed(2)} % | Inf: {latestData.inference_ms.toFixed(1)} ms
              </span>
            )}
          </div>

          {/* Alert Handlers */}
          {latestData?.anomaly ? (
            <div className="alert-box alert-danger" style={{ background: 'rgba(255, 123, 0, 0.2)', color: '#ff7b00', borderColor: '#ff7b00' }}>
              <AlertTriangle size={18} style={{ verticalAlign: 'text-bottom' }} /> HARDWARE MAINTENANCE: OPTICAL ANOMALY DETECTED
            </div>
          ) : latestData ? (
            <div className={`alert-box ${latestData.alert ? 'alert-danger' : 'alert-success'}`}>
              {latestData.alert ?
                <><AlertTriangle size={18} style={{ verticalAlign: 'text-bottom' }} /> Pol Drop Alert! Threshold breached.</> :
                <><CheckCircle2 size={18} style={{ verticalAlign: 'text-bottom' }} /> Quality Control Nominal</>
              }
            </div>
          ) : (
            <div className="alert-box" style={{ border: '1px solid var(--card-border)', color: 'var(--text-secondary)' }}>
              Standby
            </div>
          )}
        </div>

        {/* 3D Digital Twin Component */}
        <div className="card">
          <h3><Activity size={20} /> Digital Twin (Conveyor Feed Simulation)</h3>
          <div className="chart-container" style={{ background: '#0a0d14', borderRadius: '10px', overflow: 'hidden' }}>
            <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
              <SugarcaneBelt activeAlert={latestData?.alert} isRunning={isRunning} />
            </Canvas>
          </div>
        </div>
      </div>

      <div className="top-grid" style={{ gridTemplateColumns: '1fr 2fr', marginTop: '2rem' }}>
        <div className="card">
          <h3><Activity size={20} /> Live Spectrum</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={currentSpectrum}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="wavelength" stroke="#8b949e" tick={{ fill: '#8b949e' }} />
                <YAxis dataKey="absorbance" stroke="#8b949e" tick={{ fill: '#8b949e' }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }} />
                <Line type="monotone" dataKey="absorbance" stroke="#00d2ff" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3><TrendingUp size={20} /> Pol Flow Output History</h3>
          <div className="chart-container" style={{ height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={predictions}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="time" stroke="#8b949e" tick={{ fill: '#8b949e' }} />
                <YAxis domain={['auto', 'auto']} stroke="#8b949e" tick={{ fill: '#8b949e' }} />
                <Tooltip contentStyle={{ backgroundColor: '#0d1117', borderColor: '#30363d', color: '#c9d1d9' }} />
                <ReferenceLine y={threshold} label={{ position: 'top', value: 'Threshold', fill: '#ff3366', fontSize: 12 }} stroke="#ff3366" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="predicted" stroke="#09b850" strokeWidth={3} isAnimationActive={false} />
                <Line type="monotone" dataKey="actual" stroke="#ffffff" shape={<circle r={2} />} fill="#ffffff" opacity={0.5} strokeDasharray="3 3" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
