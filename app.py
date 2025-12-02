import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd

# -----------------------------
# Konfigurasi Direktori Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_depth_class.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset_gempa.csv")

# -----------------------------
# Cek file model
# -----------------------------
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("File scaler.pkl tidak ditemukan. Jalankan modeling.py terlebih dahulu.")

if not os.path.exists(XGB_PATH):
    raise FileNotFoundError("File xgb_depth_class.pkl tidak ditemukan. Jalankan modeling.py terlebih dahulu.")

# Load model dan scaler
scaler = joblib.load(SCALER_PATH)
xgb_model = joblib.load(XGB_PATH)

# -----------------------------
# Load dataset untuk membaca tahun
# -----------------------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("dataset-gempa.csv tidak ditemukan di folder project.")

df_year = pd.read_csv(DATASET_PATH)

if "year" in df_year.columns:
    year_min = int(df_year["year"].min())
    year_max = int(df_year["year"].max())
    YEARS = list(range(year_min, year_max + 1))
else:
    YEARS = [2020, 2021, 2022, 2023, 2024]  # fallback jika kolom tidak ada


# -----------------------------
# Mapping kelas
# -----------------------------
CLASS_MAP = {
    0: "Shallow (< 70 km) - Potensi kerusakan tinggi",
    1: "Intermediate (70–300 km) - Potensi kerusakan sedang",
    2: "Deep (> 300 km) - Relatif lebih aman di permukaan",
}

# Nilai default untuk fitur tidak diinput user
DEFAULT_DMIN = 0.1
DEFAULT_RMS = 0.8
DEFAULT_HERR = 5.0
DEFAULT_DERR = 5.0
DEFAULT_MAGERR = 0.1


# -----------------------------
# Fungsi Prediksi
# -----------------------------
def predict_depth_class(year, latitude, longitude, mag, gap):
    X_input = np.array([[
        year,
        latitude,
        longitude,
        mag,
        gap,
        DEFAULT_DMIN,
        DEFAULT_RMS,
        DEFAULT_HERR,
        DEFAULT_DERR,
        DEFAULT_MAGERR
    ]])

    X_scaled = scaler.transform(X_input)
    y_pred = xgb_model.predict(X_scaled)[0]

    proba = xgb_model.predict_proba(X_scaled)[0] if hasattr(xgb_model, "predict_proba") else None

    return int(y_pred), proba


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Prediksi Kedalaman Gempa", layout="wide")

# SIDEBAR (Input)
st.sidebar.title("🌋 Input Parameter Gempa")

year = st.sidebar.selectbox("Tahun Kejadian", YEARS, index=len(YEARS) - 1)

latitude = st.sidebar.slider(
    "Latitude",
    min_value=-10.0,
    max_value=10.0,
    value=-2.0,
    step=0.01
)

longitude = st.sidebar.slider(
    "Longitude",
    min_value=90.0,
    max_value=150.0,
    value=120.0,
    step=0.01
)

mag = st.sidebar.slider(
    "Magnitudo (Mw)",
    min_value=2.0,
    max_value=9.0,
    value=5.0,
    step=0.1
)

gap = st.sidebar.slider(
    "Gap",
    min_value=0.0,
    max_value=360.0,
    value=100.0,
    step=1.0
)

predict_button = st.sidebar.button("🔍 Prediksi Sekarang")

st.sidebar.caption("Parameter lain (RMS, Dmin, Error) diisi otomatis.")

# -----------------------------
# MAIN PAGE (Hasil Prediksi)
# -----------------------------
st.title("🔎 Prediksi Kelas Kedalaman Gempa Bumi")
st.write(
    """
Aplikasi ini memprediksi **kelas kedalaman gempa bumi** menggunakan model Machine Learning (XGBoost).  
Silakan isi parameter di bagian **sidebar kiri**, lalu klik **Prediksi Sekarang**.
"""
)

st.markdown("---")

# Menampilkan hasil ketika tombol diklik
if predict_button:
    y_pred, proba = predict_depth_class(year, latitude, longitude, mag, gap)

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** `{y_pred}`")
    st.write(f"**Interpretasi:** {CLASS_MAP.get(y_pred)}")

    if proba is not None:
        st.subheader("📊 Probabilitas Prediksi")
        for i, p in enumerate(proba):
            st.write(f"- **Kelas {i}** ({CLASS_MAP[i]}): **{p*100:.2f}%**")

else:
    st.info("Isi parameter di sidebar, lalu klik **Prediksi Sekarang** untuk melihat hasil.")
