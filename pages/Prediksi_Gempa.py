import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("🔍 Prediksi Kedalaman Gempa")

# Load model
@st.cache_resource
def load_all_models():
    try:
        scaler = joblib.load("models/scaler.pkl")
        xgb_model = joblib.load("models/xgb_depth_class.pkl")
        lstm_model = load_model("models/lstm_depth_class.keras")
        return scaler, xgb_model, lstm_model, None
    except Exception as e:
        return None, None, None, str(e)

scaler, xgb_model, lstm_model, err = load_all_models()

if err:
    st.error("Model tidak ditemukan! Pastikan semua file model berada di folder `models/`.")
    st.code(err)
else:
    st.success("Model berhasil dimuat.")

    st.subheader("Masukkan Parameter Gempa:")

    col1, col2, col3 = st.columns(3)

    with col1:
        latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
        longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
        mag = st.number_input("Magnitude", 0.0, 10.0, 5.0)

    with col2:
        gap = st.number_input("Gap", 0.0, 360.0, 50.0)
        dmin = st.number_input("Dmin", 0.0, 1000.0, 10.0)
        rms = st.number_input("RMS", 0.0, 10.0, 1.0)

    with col3:
        h_err = st.number_input("Horizontal Error", 0.0, 100.0, 5.0)
        d_err = st.number_input("Depth Error", 0.0, 100.0, 5.0)
        year = st.number_input("Tahun", 1900, 2100, 2023)

    if st.button("🔍 Prediksi Kedalaman Gempa"):

        features = np.array([[latitude, longitude, mag, gap, dmin, rms, h_err, d_err, year]])
        scaled = scaler.transform(features)

        lstm_input = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))

        xgb_pred = xgb_model.predict(scaled)[0]
        lstm_pred = np.argmax(lstm_model.predict(lstm_input), axis=1)[0]

        label_map = {
            0: "Shallow (<70 km)",
            1: "Intermediate (70–300 km)",
            2: "Deep (>300 km)"
        }

        st.success(f"🎯 **XGBoost Prediction:** {label_map[xgb_pred]}")
        st.info(f"🤖 **LSTM Prediction:** {label_map[lstm_pred]}")
