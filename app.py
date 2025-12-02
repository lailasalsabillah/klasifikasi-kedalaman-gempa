import os
import joblib
import numpy as np
import streamlit as st

# -----------------------------
# 1. Load scaler & model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
xgb_path    = os.path.join(BASE_DIR, "models", "xgb_depth_class.pkl")

scaler = joblib.load(scaler_path)
xgb_model = joblib.load(xgb_path)

# mapping label ke teks
CLASS_MAP = {
    0: "Shallow (< 70 km) - Paling berbahaya",
    1: "Intermediate (70–300 km) - Bahaya sedang",
    2: "Deep (> 300 km) - Relatif paling aman"
}

# -----------------------------
# 2. UI Streamlit
# -----------------------------
st.title("Klasifikasi Kedalaman Gempa Bumi (LSTM & XGBoost)")
st.write("Model ini memprediksi kelas kedalaman gempa berdasarkan fitur lokasi & parameter seismik.")

st.subheader("Input Fitur Gempa")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
    latitude = st.number_input("Latitude", value=-2.0, format="%.4f")
    longitude = st.number_input("Longitude", value=120.0, format="%.4f")
    mag = st.number_input("Magnitude (mag)", min_value=0.0, max_value=10.0, value=5.0, format="%.2f")

with col2:
    gap = st.number_input("Gap", value=100.0, format="%.2f")
    dmin = st.number_input("Dmin", value=0.1, format="%.4f")
    rms = st.number_input("RMS", value=0.8, format="%.3f")
    horizontalError = st.number_input("Horizontal Error", value=5.0, format="%.3f")
    depthError = st.number_input("Depth Error", value=5.0, format="%.3f")
    magError = st.number_input("Mag Error", value=0.1, format="%.3f")

if st.button("Prediksi Kedalaman"):
    # urutan fitur HARUS sama seperti di modeling.py
    # feature_cols = ["year", "latitude", "longitude", "mag",
    #                 "gap", "dmin", "rms", "horizontalError",
    #                 "depthError", "magError"]
    X_input = np.array([[year, latitude, longitude, mag,
                         gap, dmin, rms, horizontalError,
                         depthError, magError]])

    # scaling
    X_scaled = scaler.transform(X_input)

    # prediksi
    y_pred = xgb_model.predict(X_scaled)[0]
    label = CLASS_MAP.get(int(y_pred), "Unknown")

    st.subheader("Hasil Prediksi")
    st.write(f"**Kelas Kedalaman:** {int(y_pred)}")
    st.write(f"**Interpretasi:** {label}")
