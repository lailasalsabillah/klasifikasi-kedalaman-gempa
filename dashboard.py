import streamlit as st
import numpy as np
import joblib

# load scaler & model
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

st.title("🌋 Prediksi Kedalaman Gempa Bumi")
st.write("""
Aplikasi ini memprediksi kelas kedalaman gempa (Shallow, Intermediate, Deep)  
menggunakan model XGBoost.
""")

st.sidebar.header("Masukkan Data Gempa")

latitude = st.sidebar.number_input("Latitude", -10.0, 10.0)
longitude = st.sidebar.number_input("Longitude", 90.0, 150.0)
mag = st.sidebar.number_input("Magnitude", 3.0, 9.0)
gap = st.sidebar.number_input("Gap", 0, 300)
dmin = st.sidebar.number_input("Dmin", 0.0, 30.0)
rms = st.sidebar.number_input("RMS", 0.0, 3.0)
horizontalError = st.sidebar.number_input("Horizontal Error", 0.0, 30.0)
depthError = st.sidebar.number_input("Depth Error", 0.0, 20.0)
magError = st.sidebar.number_input("Magnitude Error", 0.0, 1.0)
year = st.sidebar.number_input("Year", 2020, 2024)

def predict(data):
    scaled = scaler.transform(data)
    pred = xgb_model.predict(scaled)[0]
    return pred

label = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

if st.sidebar.button("Prediksi"):
    data = np.array([[
        latitude, longitude, mag,
        gap, dmin, rms,
        horizontalError, depthError, magError,
        year
    ]])

    pred = predict(data)
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"**Kedalaman:** {label[pred]}")
