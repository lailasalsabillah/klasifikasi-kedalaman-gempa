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

# Daftar tahun untuk dropdown
YEARS = list(range(2020, 2024))  # 2000 - 2030

# Nilai default fitur lain (tidak diinput user)
DEFAULT_DMIN = 0.1
DEFAULT_RMS = 0.8
DEFAULT_HERR = 5.0
DEFAULT_DERR = 5.0
DEFAULT_MAGERR = 0.1


# -----------------------------
# Fungsi Prediksi
# -----------------------------
def predict_depth_class(
    year,
    latitude,
    longitude,
    mag,
    gap,
    dmin=DEFAULT_DMIN,
    rms=DEFAULT_RMS,
    horizontalError=DEFAULT_HERR,
    depthError=DEFAULT_DERR,
    magError=DEFAULT_MAGERR,
):
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

    if hasattr(xgb_model, "predict_proba"):
        proba = xgb_model.predict_proba(X_scaled)[0]
    else:
        proba = None

    return int(y_pred), proba


# -----------------------------
# UI Streamlit (Sidebar Input)
# -----------------------------
st.set_page_config(
    page_title="Klasifikasi Kedalaman Gempa Bumi", layout="wide"
)

# ============================= SIDEBAR =============================
st.sidebar.title("🌋 Input Parameter Gempa")

year = st.sidebar.selectbox("Tahun Kejadian", YEARS, index=YEARS.index(2024))

latitude = st.sidebar.slider(
    "Latitude",
    min_value=-10.0,
    max_value=10.0,
    value=-2.0,
    step=0.01,
    format="%.2f",
)

longitude = st.sidebar.slider(
    "Longitude",
    min_value=90.0,
    max_value=150.0,
    value=120.0,
    step=0.01,
    format="%.2f",
)

mag = st.sidebar.slider(
    "Magnitudo (Mw)",
    min_value=2.0,
    max_value=9.0,
    value=5.0,
    step=0.1,
    format="%.1f",
)

gap = st.sidebar.slider(
    "Gap",
    min_value=0.0,
    max_value=360.0,
    value=100.0,
    step=1.0,
    format="%.0f",
)

predict_button = st.sidebar.button("🔍 Prediksi Sekarang")

# ============================= MAIN PAGE =============================
st.title("🔎 Prediksi Kelas Kedalaman Gempa Bumi")
st.write(
    """
Aplikasi ini memprediksi **kelas kedalaman gempa bumi** berdasarkan input sederhana
(latitude, longitude, magnitudo, dan gap).  
Model yang digunakan: **XGBoost**.
"""
)

st.markdown("---")

if predict_button:
    y_pred, proba = predict_depth_class(
        year, latitude, longitude, mag, gap
    )

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** `{y_pred}`")
    st.write(f"**Interpretasi:** {CLASS_MAP.get(y_pred, 'Tidak diketahui')}")

    if proba is not None:
        st.subheader("📊 Probabilitas Tiap Kelas")
        for i, p in enumerate(proba):
            st.write(
                f"- **Kelas {i}** ({CLASS_MAP.get(i)}): **{p*100:.2f}%**"
            )

    st.info(
        "Catatan: Hasil prediksi ini merupakan output model machine learning "
        "dan bukan analisis resmi ahli seismologi."
    )

else:
    st.info("Masukkan data di sidebar, lalu klik **Prediksi Sekarang**.")
