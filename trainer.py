import pandas as pd
import numpy as np
import scipy.signal
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # We are interested in 'TS' (Total Sugar / Pol)
    # Drop rows where TS is missing
    # In some CSVs it might be empty string instead of NaN
    df['TS'] = pd.to_numeric(df['TS'], errors='coerce')
    df = df.dropna(subset=['TS'])
    
    # Extract features (wavelengths)
    # The wavelengths are columns starting with 'amplitude' or numbers.
    wavelength_cols = [col for col in df.columns if col.startswith('amplitude') or col.replace('.','',1).isdigit()]
    
    X = df[wavelength_cols].values
    y = df['TS'].values
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features (wavelengths).")
    return X, y, wavelength_cols

def preprocess_spectra(X):
    """
    Standard Chemometrics preprocessing:
    1. Savitzky-Golay filter for smoothing and 1st derivative
    2. Standard Normal Variate (SNV) to scatter-correct
    """
    # 1. Savitzky-Golay Smoothing and 1st Derivative
    # window_length=15, polyorder=2, deriv=1
    X_sg = scipy.signal.savgol_filter(X, window_length=15, polyorder=2, deriv=1)
    
    # 2. Standard Normal Variate (SNV)
    # For each spectrum, subtract mean and divide by standard deviation
    mean = np.mean(X_sg, axis=1, keepdims=True)
    std = np.std(X_sg, axis=1, keepdims=True)
    X_snv = (X_sg - mean) / (std + 1e-8)
    
    return X_snv

def train_pls(X_train, y_train):
    print("Finding optimal number of PLS components via Cross-Validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_n = 1
    lowest_mse = float('inf')
    
    # Test up to 15 components
    for i in range(1, 15):
        pls = PLSRegression(n_components=i)
        score = -cross_val_score(pls, X_train, y_train, cv=cv, scoring='neg_mean_squared_error').mean()
        if score < lowest_mse:
            lowest_mse = score
            best_n = i
            
    print(f"Best number of PLS components: {best_n} (CV MSE: {lowest_mse:.4f})")
    
    pls = PLSRegression(n_components=best_n)
    pls.fit(X_train, y_train)
    return pls

def train_ann(X_train, y_train):
    print("Training Artificial Neural Network (MLPRegressor)...")
    # A simple architecture since dataset is likely small
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True)
    ann.fit(X_train, y_train)
    return ann

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # Output to console simulating mock industrial output readiness
    print(f"--- {name} Evaluation ---")
    print(f"RMSEP (Root Mean Square Error of Prediction): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return rmse, r2

def main():
    dataset_path = "dataset/Scio.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please ensure it is extracted to dataset/Scio.csv")
        return
        
    X, y, wavelengths = load_data(dataset_path)
    
    # Save wavelengths for the simulation dashboard
    joblib.dump(wavelengths, "wavelengths.pkl")
    
    # Preprocess
    print("Preprocessing spectra (Savitzky-Golay + SNV)...")
    X_prep = preprocess_spectra(X)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
    
    # Standardize Targets (optional but good for comparison)
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))
    
    # Train PLS
    pls = train_pls(X_train, y_train)
    evaluate_model(pls, X_test, y_test, "PLS Regression")
    
    # Train ANN (Optional but requested for comparison)
    ann = train_ann(X_train, y_train)
    evaluate_model(ann, X_test, y_test, "Artificial Neural Network")
    
    # Save the best model (typically PLS for NIR, widely used in sugarcane mills)
    # We will save the PLS model for our real-time dashboard simulation
    print("Saving PLS model to pls_model.pkl...")
    joblib.dump(pls, 'pls_model.pkl')
    
    # Save test set for simulation later
    np.save('X_test_raw.npy', X[len(X_train):])
    np.save('y_test.npy', y_test)
    
    print("Training pipeline complete.")

if __name__ == "__main__":
    main()
