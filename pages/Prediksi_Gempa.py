import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import base64

st.title("🔎 Prediksi Kedalaman Gempa")

# Load model
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")
lstm_model = load_model("models/lstm_depth_class.keras")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

# Input user
with st.sidebar:
    st.header("Masukkan Parameter Gempa (Pilih Rentang)")

    lat = st.slider("Latitude", -20.0, 20.0, (-10.0, 10.0))
    lon = st.slider("Longitude", 80.0, 150.0, (100.0, 120.0))
    mag = st.slider("Magnitude", 3.0, 10.0, (4.0, 6.0))
    gap = st.slider("Gap", 0, 300, (20, 80))
    dmin = st.slider("Dmin", 0.0, 30.0, (1.0, 5.0))
    rms = st.slider("RMS", 0.0, 3.0, (0.5, 1.0))
    herror = st.slider("Horizontal Error", 0.0, 50.0, (5.0, 10.0))
    derror = st.slider("Depth Error", 0.0, 30.0, (3.0, 8.0))
    magerr = st.slider("Magnitude Error", 0.0, 1.0, (0.05, 0.2))
    year = st.slider("Year", 2000, 2030, (2015, 2025))

    btn = st.button("Prediksi Gempa")

if btn:
    # Gunakan nilai rata-rata dari rentang
    lat_val = np.mean(lat)
    lon_val = np.mean(lon)
    mag_val = np.mean(mag)
    gap_val = np.mean(gap)
    dmin_val = np.mean(dmin)
    rms_val = np.mean(rms)
    herror_val_
