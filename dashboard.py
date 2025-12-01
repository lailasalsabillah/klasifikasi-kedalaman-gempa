# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    layout="wide",
    page_icon="🌋"
)

# ============================
# LOAD MODEL
# ============================
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/xgb_depth_class.pkl")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

danger_map = {
    0: ("Bahaya Tinggi", "red"),
    1: ("Bahaya Sedang", "orange"),
    2: ("Bahaya Rendah", "blue")
}

# ============================
# SIDEBAR INPUT
# ============================
st.sidebar.title("🔍 Input Parameter Gempa Bumi")

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

predict_btn = st.sidebar.button("🔎 Prediksi Kedalaman Gempa")

# ============================
# FUNCTION PREDICT
# ============================
def predict_depth():
    data = np.array([[
        latitude, longitude, mag, gap, dmin, rms,
        horizontalError, depthError, magError, year
    ]])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    return pred, data

# ============================
# TITLE
# ============================
st.title("🌋 Prediksi Kedalaman Gempa Bumi")
st.write("Aplikasi prediksi kedalaman gempa berdasarkan parameter seismik menggunakan model **XGBoost**.")

st.markdown("---")

# ============================
# RESULT PAGE
# ============================
if predict_btn:

    pred, data = predict_depth()
    depth_label = label_map[pred]
    bahaya, color = danger_map[pred]

    # Convert input to DataFrame
    df_input = pd.DataFrame(data, columns=[
        "Latitude", "Longitude", "Magnitude", "Gap", "Dmin",
        "RMS", "HorizontalError", "DepthError", "MagError", "Year"
    ])

    # ============================
    # 1. CARD HASIL PREDIKSI
    # ============================
    st.subheader("📊 Hasil Prediksi Kedalaman Gempa")

    st.markdown(
        f"""
        <div style="
            padding: 22px; 
            border-radius: 12px; 
            background-color: #f8f9fa;
            border-left: 12px solid {color};">
            <h3 style="margin: 0; color:{color};">{depth_label}</h3>
            <p style="font-size: 16px; margin-top: 8px;">
                (Prediksi berdasarkan model <b>XGBoost</b>)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ============================
    # 2. GRAFIK VISUALISASI
    # ============================
    st.subheader("📈 Visualisasi Magnitude vs Tingkat Bahaya")

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Magnitude"], [mag], color=color)
    ax.set_ylabel("Magnitude (M)")
    ax.set_title("Magnitude dari Gempa yang Diprediksi")
    st.pyplot(fig)

    st.markdown("---")

    # ============================
    # 3. TABEL PARAMETER INPUT
    # ============================
    st.subheader("🧾 Tabel Parameter Gempa yang Dimasukkan")
    st.dataframe(df_input, use_container_width=True)

    # ============================
    # 4. INFORMASI TAMBAHAN
    # ============================
    st.markdown("### ℹ Penjelasan Kategori Kedalaman Gempa")
    st.write("""
    **Shallow (<70 km)** → Sangat berbahaya karena energi belum banyak meredam.  
    **Intermediate (70–300 km)** → Bahaya sedang dan masih dirasakan di wilayah luas.  
    **Deep (>300 km)** → Energi sudah banyak teredam sehingga bahaya lebih rendah.
    """)

else:
    st.info("Masukkan parameter di sidebar, lalu klik **Prediksi Kedalaman Gempa**.")
