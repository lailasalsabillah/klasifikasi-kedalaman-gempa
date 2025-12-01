import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")
lstm_model = load_model("models/lstm_depth_class.keras")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

st.title("⚡ Prediksi Kedalaman Gempa")

st.write("Isi parameter berikut untuk memprediksi kelas kedalaman gempa:")

# Input
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", value=-6.5)
    longitude = st.number_input("Longitude", value=107.0)
    mag = st.number_input("Magnitude (Mag)", value=4.5)
    gap = st.number_input("Gap", value=80)

with col2:
    dmin = st.number_input("Dmin", value=2.1)
    rms = st.number_input("RMS", value=0.55)
    horizontalError = st.number_input("Horizontal Error", value=8.0)
    depthError = st.number_input("Depth Error", value=6.0)
    magError = st.number_input("Magnitude Error", value=0.12)
    year = st.number_input("Tahun", value=2023)

if st.button("🔍 Prediksi Kedalaman Gempa"):
    # Data array
    data = np.array([[latitude, longitude, mag, gap, dmin, rms,
                      horizontalError, depthError, magError, year]])

    scaled = scaler.transform(data)
    lstm_input = scaled.reshape((1, 1, scaled.shape[1]))

    pred_xgb = xgb_model.predict(data)[0]
    pred_lstm = np.argmax(lstm_model.predict(lstm_input), axis=1)[0]

    st.subheader("📌 Hasil Prediksi")
    st.success(f"**XGBoost:** {label_map[pred_xgb]}")
    st.info(f"**LSTM:** {label_map[pred_lstm]}")
