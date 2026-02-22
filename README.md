# 🌾 Sugarcane NIR Real-Time Pol Predictor (Hackathon Prototype)

This repository contains a full, pure software prototype for real-time, non-intrusive prediction of sugar percentage (Pol / Total Sugar) in incoming sugarcane on a conveyor belt using Near-Infrared (NIR) spectroscopy.

```text
+----------------+      +----------------+      +-----------------+
| NIR Sensor     |      | Edge Device    |      | SCADA / Mill DB |
| (Spectrometer) | ---> | (Raspberry Pi) | ---> | (MQTT / OPC-UA) |
+-------+--------+      +-------+--------+      +-----------------+
        |                       |    
        v                       v
 [Optical Feed]         [PLS ML Engine]
 Sugarcane Conveyor     Model Prediction
```

## 🚀 Quickstart

1. **Install Dependencies (if not already done)**: `pip install -r requirements.txt`
2. **Train Model (already done)**: `python trainer.py`
3. **Run Dashboard**: Double-click `run_demo.bat` OR run `streamlit run app.py`.

## 🧠 Why NIR + PLS?
To measure sugar content (Pol) fast, accurately, and without destroying the sample, **Near-Infrared (NIR)** spectroscopy is utilized. 
However, raw NIR data has a massive number of overlapping features (wavelengths). **Partial Least Squares (PLS) Regression** solves this problem by projecting those features into a smaller, highly predictive component space.
- **Fast**: Inference is in milliseconds.
- **Robust**: Easily runs on cheap edge-devices like PLCs or Raspberry Pi.
- **Proven**: It is the industry gold-standard for chemometrics.

## 📊 Accuracy Backed by Literature
Literature frequently cites that calibrated NIR spectrometers paired with preprocessed chemometric models (like we've implemented with Savitzky-Golay + SNV + PLS) achieve an **RMSEP consistently ranging between 0.2% and 1.0%** for sugarcane Pol, making it highly feasible for this prototype to scale in actual mills.

## ⚙️ How to Improve Further (Future Scope)
* **More Data**: Integrate spectra from multiple harvests and ambient temperatures to increase generalizability.
* **Site-Specific Calibration**: Localized bias correction for the specific sugarcane clones used universally at a given mill site.
* **Continuous Learning Engine**: Auto-update the PLS components when actual lab samples periodically align (e.g. daily quality assurance testing).
