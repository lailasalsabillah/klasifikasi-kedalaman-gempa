# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Klasifikasi Kedalaman Gempa Bumi",
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
    0: ("Tinggi", "red"),
    1: ("Sedang", "orange"),
    2: ("Rendah", "blue")
}

# ============================
# SIDEBAR INPUT  
# ============================
st.sidebar.title("🔍 Input Parameter Gempa")

latitude = st.sidebar.number_input("Latitude", -12.0, 10.0, -2.0)
longitude = st.sidebar.number_input("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.number_input("Magnitudo", 3.0, 9.0, 4.5)
gap = st.sidebar.number_input("Gap", 0, 300, 80)
dmin = st.sidebar.number_input("Dmin", 0.0, 30.0, 2.1)
rms = st.sidebar.number_input("RMS", 0.0, 3.0, 0.7)
horizontalError = st.sidebar.number_input("Horizontal Error", 0.0, 30.0, 8.0)
depthError = st.sidebar.number_input("Depth Error", 0.0, 20.0, 6.0)
magError = st.sidebar.number_input("Magnitude Error", 0.0, 1.0, 0.12)
year = st.sidebar.number_input("Tahun", 2020, 2024, 2023)

predict_btn = st.sidebar.button("Prediksi Kedalaman Gempa")

# ============================
# TITLE
# ============================

st.title("🌋 Klasifikasi Kedalaman Gempa Bumi")
st.write("Model prediksi kedalaman berdasarkan parameter seismik menggunakan algoritma **XGBoost**.")

st.markdown("---")

# ============================
# FUNCTION PREDICT
# ============================
def predict():
    data = np.array([[
        latitude, longitude, mag, gap, dmin, rms,
        horizontalError, depthError, magError, year
    ]])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    return pred

# ============================
# SHOW PREDICTION
# ============================
if predict_btn:

    pred = predict()
    depth_label = label_map[pred]
    bahaya, color = danger_map[pred]

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📊 Hasil Prediksi")
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 15px;
                background-color: #f8f9fa;
                border-left: 10px solid {color};">
                
                <h3 style="margin-bottom: 5px;">{depth_label}</h3>
                <p style="font-size: 18px;">Tingkat Bahaya: 
                    <b style="color:{color};">{bahaya}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📈 Visualisasi Bahaya Berdasarkan Kedalaman")
        st.progress(100 if pred == 0 else (60 if pred == 1 else 30))

    # ============================
    # MAP LOCATION
    # ============================
    with col2:
        st.subheader("🗺️ Lokasi Gempa")
        map_center = [latitude, longitude]
        m = folium.Map(location=map_center, zoom_start=6)

        folium.Marker(
            location=map_center,
            popup="Lokasi Gempa",
            icon=folium.Icon(color=color)
        ).add_to(m)

        st_folium(m, height=350)

    st.markdown("---")
    
    st.subheader("ℹ Penjelasan Kategori Kedalaman")
    st.write("""
    - **Shallow (<70 km)** → Wilayah paling berbahaya karena energinya belum mereda.  
    - **Intermediate (70–300 km)** → Bahaya sedang, masih bisa dirasakan luas.  
    - **Deep (>300 km)** → Energi banyak teredam, bahaya rendah.  
    """)

else:
    st.info("Masukkan parameter pada sidebar lalu klik **Prediksi Kedalaman Gempa**.")


