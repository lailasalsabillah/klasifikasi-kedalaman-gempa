import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CUSTOM CSS – TEMA TECTONIC MODERN
# ============================================================
st.markdown("""
<style>

:root {
    --primary-color: #1abc9c;
    --secondary-color: #16a085;
    --background: #f4fefb;
    --card-bg: #ffffff;
    --card-border: #d9f7ef;
    --text-dark: #0a3d3f;
    --radius: 18px;
}

/* Background */
.stApp {
    background-color: var(--background);
}

/* Card Style */
.card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: var(--radius);
    border: 1.5px solid var(--card-border);
    box-shadow: 0px 4px 14px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Header Title */
.header-title {
    font-size: 38px;
    font-weight: 800;
    color: var(--text-dark);
    text-align: center;
    margin-bottom: 5px;
}

/* Subheader subtitle */
.header-sub {
    font-size: 18px;
    text-align: center;
    color:#198f84;
    margin-bottom: 25px;
}

/* Styled Divider */
.divider {
    height: 3px;
    background: linear-gradient(90deg,var(--primary-color),transparent);
    margin: 15px 0;
    border-radius: 20px;
}

/* Buttons */
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

/* Result Badge */
.badge {
    display:inline-block;
    padding:6px 14px;
    border-radius:14px;
    color:white;
    font-weight:600;
}
.badge-0 { background:#e74c3c; }
.badge-1 { background:#f1c40f; }
.badge-2 { background:#27ae60; }

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
# FUNGSI PREDIKSI
# ============================================================
def predict_depth(year, lat, lon, mag, gap, dmin, rms, herr, magerr):
    X = np.array([[year, lat, lon, mag, gap, dmin, rms, herr, magerr]])
    X_scaled = scaler.transform(X)

    pred = xgb_model.predict(X_scaled)[0]
    proba = xgb_model.predict_proba(X_scaled)[0]
    return pred, proba


# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown("<div class='header-title'>🌋 Klasifikasi Kedalaman Gempa</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>XGBoost Earthquake Depth Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR BARU — FITUR, SUMBER DATA, WARNING, DEVELOPER
# ============================================================

st.sidebar.markdown("## ✨ Fitur:")
st.sidebar.markdown("""
- 🗺️ **Peta interaktif**
- 📊 **Filter magnitudo**
- 📋 **Tabel data lengkap**
- 📥 **Download data CSV**
- 🚨 **Peringatan gempa besar**
""")

st.sidebar.markdown("---")

# Sumber Data
st.sidebar.markdown("## 📌 Sumber Data:")
st.sidebar.markdown("""
- 🌎 **USGS (United States Geological Survey)**
- 🇮🇩 Area: **Indonesia**
- 🔄 Update: **Realtime**
""")

st.sidebar.markdown("---")

# Input Parameter Gempa
st.sidebar.markdown("## 🌍 Input Data Gempa")

year = st.sidebar.selectbox("Tahun", YEARS)
lat  = st.sidebar.slider("Latitude",  -12.0, 8.0, -2.0)
lon  = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0)
mag  = st.sidebar.slider("Magnitudo", 2.0, 9.0, 5.0)
gap  = st.sidebar.slider("Gap", 0, 360, 100)
dmin = st.sidebar.slider("Dmin", 0.0, 20.0, 5.0)
rms  = st.sidebar.slider("RMS", 0.0, 2.0, 0.5)
herr = st.sidebar.slider("Horizontal Error", 0.0, 30.0, 5.0)
magerr = st.sidebar.slider("Magnitude Error", 0.0, 0.5, 0.05)

btn = st.sidebar.button("🔍 Prediksi Sekarang")

st.sidebar.markdown("---")

# Sistem Warning
if mag >= 6.0:
    st.sidebar.markdown("""
    <div style='padding:14px; background:#ffdddd; border-left:6px solid red; 
                border-radius:10px; margin-bottom:10px;'>
        <b>🚨 PERINGATAN GEMPA BESAR!</b><br>
        Magnitudo ≥ 6.0 terdeteksi dari input.
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    <div style='padding:14px; background:#fff6d5; border-left:6px solid #f1c40f; 
                border-radius:10px; margin-bottom:10px;'>
        <b>ℹ️ Tidak ada gempa besar.</b><br>
        Magnitudo input: {mag}
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Developer Box
st.sidebar.markdown("""
<div style='padding:14px; background:#e8f9f5; border-radius:12px; text-align:center;'>
    <b>👩‍💻 Dibuat oleh:</b><br>
    <span style='font-size:17px; color:#0a3d3f;'>
        <b>Laila Salsabillah</b>
    </span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Hasil Prediksi", "📊 Grafik", "📁 Info Dataset"])


# ============================================================
# TAB 1 — HASIL PREDIKSI
# ============================================================
with tab1:
    if btn:
        pred, proba = predict_depth(year, lat, lon, mag, gap, dmin, rms, herr, magerr)

        # RINGKASAN INPUT
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📝 Ringkasan Input Pengguna")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- **Tahun:** {year}")
            st.write(f"- **Latitude:** {lat}")
            st.write(f"- **Longitude:** {lon}")
            st.write(f"- **Magnitudo:** {mag}")
        with col2:
            st.write(f"- **Gap:** {gap}")
            st.write(f"- **Dmin:** {dmin}")
            st.write(f"- **RMS:** {rms}")
            st.write(f"- **Horizontal Error:** {herr}")
            st.write(f"- **Magnitude Error:** {magerr}")

        st.markdown("</div>", unsafe_allow_html=True)

        # HASIL PREDIKSI
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 Hasil Prediksi Kedalaman Gempa")

        st.markdown(
            f"<span class='badge badge-{pred}' style='font-size:20px;'>Kelas {pred}</span>",
            unsafe_allow_html=True
        )

        st.write(f"**Interpretasi:** {CLASS_MAP[pred]}")
        st.markdown("</div>", unsafe_allow_html=True)

        # PROBABILITAS
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Probabilitas Prediksi")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        for i, p in enumerate(proba):
            st.write(f"**Kelas {i}** — {CLASS_MAP[i]}: `{p*100:.2f}%`")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("👈 Masukkan input, lalu klik *Prediksi Sekarang*.")


# ============================================================
# TAB 2 — GRAFIK
# ============================================================
with tab2:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Probabilitas Kelas (Plotly)")

    if btn:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Kelas 0", "Kelas 1", "Kelas 2"],
            y=proba,
            marker=dict(color=["#e74c3c", "#f1c40f", "#27ae60"])
        ))
        fig.update_layout(title="Probabilitas Prediksi")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Grafik muncul setelah melakukan prediksi.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 3 — INFO DATASET
# ============================================================
with tab3:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📁 Informasi Dataset")

    st.write("Jumlah data:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Tahun unik:", list(df["year"].unique()))

    st.dataframe(df.head())

    st.markdown("</div>", unsafe_allow_html=True)
