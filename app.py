import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CUSTOM CSS – PALET WARNA NOMOR 2 (EARTHQUAKE THEME)
# ============================================================
st.markdown("""
<style>

:root {
    --primary-color: #E63946;      
    --secondary-color: #457B9D;    
    --background: #F8F9FA;         
    --card-bg: #FFFFFF;
    --card-border: #DDE5EC;
    --text-dark: #1D3557;          
    --radius: 18px;
}

.stApp { background-color: var(--background); }

.card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: var(--radius);
    border: 1.5px solid var(--card-border);
    box-shadow: 0px 4px 14px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.header-title {
    font-size: 38px;
    font-weight: 800;
    color: var(--text-dark);
    text-align: center;
    margin-bottom: 5px;
}

.header-sub {
    font-size: 18px;
    text-align: center;
    color: var(--secondary-color);
    margin-bottom: 25px;
}

.divider {
    height: 3px;
    background: linear-gradient(90deg,var(--primary-color),transparent);
    margin: 15px 0;
    border-radius: 20px;
}

.stButton button {
    background-color: var(--primary-color);
    color:white;
    border-radius:var(--radius);
    padding:10px 16px;
    border:none;
    transition:0.2s;
}
.stButton button:hover {
    background-color: var(--secondary-color);
    transform:scale(1.03);
}

.badge {
    display:inline-block;
    padding:6px 14px;
    border-radius:14px;
    color:white;
    font-weight:600;
}
.badge-0 { background:#E63946; }
.badge-1 { background:#F1C40F; }
.badge-2 { background:#2ECC71; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & DATASET
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_depth_class.pkl"))

df = pd.read_csv(os.path.join(BASE_DIR, "dataset-gempa.csv"))
YEARS = sorted(df["year"].unique())

CLASS_MAP = {
    0: "Shallow (< 70 km) – Sangat Berpotensi",
    1: "Intermediate (70–300 km)",
    2: "Deep (> 300 km) – Relatif Aman"
}


# ============================================================
# FUNGSI PREDIKSI MODEL
# ============================================================
def predict_depth(year, lat, lon, depth, gap, dmin, rms, herr, magerr):
    X = np.array([[year, lat, lon, depth, gap, dmin, rms, herr, magerr]])
    X_scaled = scaler.transform(X)

    pred = xgb_model.predict(X_scaled)[0]
    proba = xgb_model.predict_proba(X_scaled)[0]
    return pred, proba


# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown("<div class='header-title'>🌋 Klasifikasi Kedalaman Gempa</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>LSTM & XGBoost Earthquake Depth Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ✨ Fitur:")
st.sidebar.markdown("""
- 📊 Grafik Kedalaman  
- 📋 Tabel Data  
- 📥 Download CSV  
- 🚨 Peringatan Gempa Dangkal  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("## 📌 Sumber Data:")
st.sidebar.markdown("""
- 🌎 USGS  
- 🇮🇩 Indonesia Region  
- 🔄 Informasi Gempa  
""")

st.sidebar.markdown("---")

# INPUT
st.sidebar.markdown("## 🌍 Input Data Gempa")

year = st.sidebar.selectbox("Tahun", YEARS)
lat  = st.sidebar.slider("Latitude",  -12.0, 8.0, -2.0)
lon  = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0)
depth = st.sidebar.slider("Kedalaman (km)", 0.0, 700.0, 10.0)
gap  = st.sidebar.slider("Gap", 0, 360, 100)
dmin = st.sidebar.slider("Dmin", 0.0, 20.0, 5.0)
rms  = st.sidebar.slider("RMS", 0.0, 2.0, 0.5)
herr = st.sidebar.slider("Horizontal Error", 0.0, 30.0, 5.0)
magerr = st.sidebar.slider("Magnitude Error", 0.0, 0.5, 0.05)

btn = st.sidebar.button("🔍 Prediksi Sekarang")

st.sidebar.markdown("---")

# WARNING
if depth < 70:
    st.sidebar.markdown("""
    <div style='padding:14px; background:#FDECEA; border-left:6px solid #E63946; border-radius:10px;'>
        <b>🚨 GEMPA DANGKAL!</b><br>Kedalaman < 70 km terdeteksi.
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    <div style='padding:14px; background:#FFF9DB; border-left:6px solid #F1C40F; border-radius:10px;'>
        <b>ℹ️ Kedalaman Aman</b><br>Kedalaman: {depth} km
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='padding:14px; background:#E8F2FB; border-radius:12px; text-align:center;'>
    <b>👩‍💻 Dibuat oleh:</b><br>
    <b>Laila Salsabilla Hanifa – 202210715333</b>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TAB NAVIGASI 4 TAB
# ============================================================
tab_info, tab_pred, tab_graph, tab_dataset = st.tabs([
    "🌍 Informasi Gempa",
    "🔮 Hasil Prediksi",
    "📊 Grafik",
    "📁 Info Dataset"
])
