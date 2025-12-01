import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load model
model = joblib.load("models/xgb_depth_class.pkl")

st.title("Klasifikasi Kedalaman Gempa")

st.write("Aplikasi prediksi kedalaman gempa (Shallow, Intermediate, Deep) menggunakan model XGBoost.")

# Input user
year = st.number_input("Tahun", min_value=2020, max_value=2024, value=2023)
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
mag = st.number_input("Magnitudo")
gap = st.number_input("Gap")
dmin = st.number_input("Dmin")
rms = st.number_input("RMS")
horizontalError = st.number_input("Horizontal Error")
depthError = st.number_input("Depth Error")
magError = st.number_input("Magnitude Error")

if st.button("Prediksi"):
    data = pd.DataFrame({
        "year": [year],
        "latitude": [latitude],
        "longitude": [longitude],
        "mag": [mag],
        "gap": [gap],
        "dmin": [dmin],
        "rms": [rms],
        "horizontalError": [horizontalError],
        "depthError": [depthError],
        "magError": [magError]
    })
    
    pred = model.predict(data)[0]
    
    mapping = {0:"Shallow (<70 km)", 1:"Intermediate (70–300 km)", 2:"Deep (>300 km)"}
    
    st.success(f"Hasil Prediksi: **{mapping[pred]}**")
