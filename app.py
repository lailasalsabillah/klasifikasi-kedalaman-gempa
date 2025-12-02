import os
import numpy as np
import joblib
import streamlit as st

# -----------------------------
# Konfigurasi & Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_depth_class.pkl")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaler tidak ditemukan di: {SCALER_PATH}. "
        f"Jalankan dulu modeling.py untuk melatih dan menyimpan model."
    )

if not os.path.exists(XGB_PATH):
    raise FileNotFoundError(
        f"Model XGBoost tidak ditemukan di: {XGB_PATH}. "
        f"Jalankan dulu modeling.py untuk melatih dan menyimpan model."
    )

scaler = joblib.load(SCALER_PATH)
xgb_model = joblib.load(XGB_PATH)

# Mapping kelas ke teks
CLASS_MAP = {
    0: "Shallow (< 70 km) - Potensi kerusakan tinggi",
    1: "Intermediate (70–300 km) - Potensi kerusakan sedang",
    2: "Deep (> 300 km) - Umumnya lebih aman di permukaan",
}

FEATURE_COLS = [
    "year",
    "latitude",
    "longitude",
    "mag",
    "gap",
    "dmin",
    "rms",
    "horizontalError",
    "depthError",
    "magError",
]


# -----------------------------
# Fungsi Prediksi
# -----------------------------
def predict_depth_class(
    year,
    latitude,
    longitude,
    mag,
    gap,
    dmin,
    rms,
    horizontalError,
    depthError,
    magError,
):
    """
    Melakukan prediksi kelas kedalaman gempa menggunakan model XGBoost.
    """
    X_input = np.array(
        [
            [
                year,
                latitude,
                longitude,
                mag,
                gap,
                dmin,
                rms,
                horizontalError,
                depthError,
                magError,
            ]
        ]
    )

    X_scaled = scaler.transform(X_input)
    y_pred = xgb_model.predict(X_scaled)[0]

    # Kalau model dilatih dengan objective multi:softprob
    if hasattr(xgb_model, "predict_proba"):
        proba = xgb_model.predict_proba(X_scaled)[0]
    else:
        proba = None

    return int(y_pred), proba


# -----------------------------
# UI Streamlit
# -----------------------------
st.set_page_config(
    page_title="Klasifikasi Kedalaman Gempa Bumi", layout="centered"
)

st.title("🌋 Klasifikasi Kedalaman Gempa Bumi")
st.write(
    """
Aplikasi ini memprediksi **kelas kedalaman gempa bumi** berdasarkan 
fitur-fitur seismik menggunakan model **XGBoost** yang telah dilatih.

Kelas kedalaman:
- **0** : Shallow (< 70 km)  
- **1** : Intermediate (70–300 km)  
- **2** : Deep (> 300 km)  
"""
)

st.markdown("---")

st.subheader("Input Parameter Gempa")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Tahun", min_value=1900, max_value=2100, value=2024)
    latitude = st.number_input("Latitude", value=-2.0, format="%.4f")
    longitude = st.number_input("Longitude", value=120.0, format="%.4f")
    mag = st.number_input(
        "Magnitudo (Mw)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        format="%.2f",
    )
    gap = st.number_input("Gap", value=100.0, format="%.2f")

with col2:
    dmin = st.number_input("Dmin", value=0.1, format="%.4f")
    rms = st.number_input("RMS", value=0.8, format="%.3f")
    horizontalError = st.number_input(
        "Horizontal Error", value=5.0, format="%.3f"
    )
    depthError = st.number_input("Depth Error", value=5.0, format="%.3f")
    magError = st.number_input("Mag Error", value=0.1, format="%.3f")

st.markdown("---")

if st.button("🔍 Prediksi Kedalaman Gempa"):
    y_pred, proba = predict_depth_class(
        year,
        latitude,
        longitude,
        mag,
        gap,
        dmin,
        rms,
        horizontalError,
        depthError,
        magError,
    )

    st.subheader("Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** `{y_pred}`")
    st.write(f"**Interpretasi:** {CLASS_MAP.get(y_pred, 'Tidak diketahui')}")

    if proba is not None:
        st.write("**Probabilitas Tiap Kelas:**")
        for i, p in enumerate(proba):
            st.write(
                f"- Kelas {i} ({CLASS_MAP.get(i, '')}): **{p * 100:.2f}%**"
            )

    st.info(
        "Catatan: Hasil prediksi ini merupakan output model machine learning "
        "dan tidak menggantikan analisis resmi dari ahli seismologi."
    )
