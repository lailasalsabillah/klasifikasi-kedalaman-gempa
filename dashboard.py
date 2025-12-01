# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import joblib

# ==================================
# CONFIGURATIONS
# ==================================
st.set_page_config(page_title="Earthquake Depth Classification", layout="wide")

# ==================================
# CSS (Background, Buttons, Styles)
# ==================================
home_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://raw.githubusercontent.com/lailasalsabillah/earthquake-depth-classification/main/background_gempa.jpeg');
    background-size: cover;
    background-position: center;
}
</style>
"""

result_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: #ffffff !important;
}
</style>
"""

# Style card hasil
st.markdown("""
<style>
.result-card {
    background: #f8f9fa;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dee2e6;
}
.pred-label {
    font-size: 22px;
    font-weight: 700;
}
.small-note {
    color: #6c757d;
    font-size: 14px;
}
.back-button {
    padding: 10px 20px;
    background: #198754;
    color: white !important;
    border-radius: 10px;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# ==================================
# SESSION STATE
# ==================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ==================================
# LOAD MODEL
# ==================================
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/xgb_depth_class.pkl")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

# ==================================
# SIDEBAR INPUT
# ==================================
st.sidebar.header("Masukkan Parameter Gempa")

latitude = st.sidebar.number_input("Latitude", -12.0, 10.0, -2.0)
longitude = st.sidebar.number_input("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.number_input("Magnitude", 3.0, 9.0, 4.5)
gap = st.sidebar.number_input("Gap", 0, 300, 80)
dmin = st.sidebar.number_input("Dmin", 0.0, 30.0, 2.1)
rms = st.sidebar.number_input("RMS", 0.0, 3.0, 0.7)
horizontalError = st.sidebar.number_input("Horizontal Error", 0.0, 30.0, 8.0)
depthError = st.sidebar.number_input("Depth Error", 0.0, 20.0, 6.0)
magError = st.sidebar.number_input("Magnitude Error", 0.0, 1.0, 0.12)
year = st.sidebar.number_input("Year", 2020, 2024, 2023)

# tombol prediksi
if st.sidebar.button("🔍 Prediksi Kedalaman Gempa ➜"):
    st.session_state.page = "result"

# ==================================
# FUNCTION PREDICT
# ==================================
def predict_depth():
    data = np.array([[
        latitude, longitude, mag, gap, dmin, rms,
        horizontalError, depthError, magError, year
    ]])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    return pred

# ==================================
# PAGE 1: HOMEPAGE
# ==================================
if st.session_state.page == "home":
    st.markdown(home_bg, unsafe_allow_html=True)

    st.markdown("""
        <div style="background: rgba(255,255,255,0.9);
                    padding: 40px;
                    border-radius: 20px;
                    margin-top: 100px;
                    text-align: center;">
            <h1><b>Klasifikasi Kedalaman Gempa Bumi</b> 🌋</h1>
            <p>Aplikasi ini memprediksi apakah gempa tergolong 
            <b>Shallow</b>, <b>Intermediate</b>, atau <b>Deep</b>
            berdasarkan parameter seismik.</p>
            <p>Masukkan data pada sidebar, lalu klik tombol prediksi.</p>
        </div>
    """, unsafe_allow_html=True)

# ==================================
# PAGE 2: HASIL PREDIKSI
# ==================================
elif st.session_state.page == "result":
    st.markdown(result_bg, unsafe_allow_html=True)
    st.header("📊 Hasil Prediksi Kedalaman Gempa")

    pred = predict_depth()

    color = "#dc3545" if pred == 0 else ("#ffc107" if pred == 1 else "#0d6efd")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="pred-label" style="color:{color};">
                {label_map[pred]}
            </div>
            <p class="small-note">(Prediksi berdasarkan model XGBoost)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
