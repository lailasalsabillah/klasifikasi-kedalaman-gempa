import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    raise FileNotFoundError("dataset-gempa.csv tidak ditemukan.")

df_year = pd.read_csv(DATASET_PATH)

if "year" in df_year.columns:
    year_min = int(df_year["year"].min())
    year_max = int(df_year["year"].max())
    YEARS = list(range(year_min, year_max + 1))
else:
    YEARS = [2020, 2021, 2022, 2023, 2024]  # fallback

# -----------------------------
# Mapping kelas
# -----------------------------
CLASS_MAP = {
    0: "Shallow (< 70 km) - Berpotensi tinggi",
    1: "Intermediate (70–300 km) - Sedang",
    2: "Deep (> 300 km) - Relatif paling aman",
}

# Nilai default fitur lain
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

# SIDEBAR INPUT
st.sidebar.title("🌋 Input Parameter Gempa")

year = st.sidebar.selectbox("Tahun Kejadian", YEARS, index=len(YEARS) - 1)

latitude = st.sidebar.slider("Latitude", -10.0, 10.0, -2.0, 0.01)
longitude = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0, 0.01)
mag = st.sidebar.slider("Magnitudo (Mw)", 2.0, 9.0, 5.0, 0.1)
gap = st.sidebar.slider("Gap", 0.0, 360.0, 100.0, 1.0)

predict_button = st.sidebar.button("🔍 Prediksi Sekarang")

st.sidebar.caption("Parameter lain diisi otomatis (Dmin, RMS, Error).")

# MAIN PAGE
st.title("🔎 Prediksi Kelas Kedalaman Gempa Bumi")
st.markdown("---")

# ======================= HASIL PREDIKSI ==========================
if predict_button:
    y_pred, proba = predict_depth_class(year, latitude, longitude, mag, gap)

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** `{y_pred}`")
    st.write(f"**Interpretasi:** {CLASS_MAP[y_pred]}")

    # ======================= BAR CHART ==========================
    if proba is not None:
        st.subheader("📊 Grafik Probabilitas Kelas")

        fig, ax = plt.subplots(figsize=(6, 4))
        kelas = ["Kelas 0", "Kelas 1", "Kelas 2"]
        colors = ["#ff6b6b", "#feca57", "#1dd1a1"]

        ax.bar(kelas, proba, color=colors)
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        ax.set_title("Probabilitas Model untuk Setiap Kelas")

        for i, p in enumerate(proba):
            ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center")

        st.pyplot(fig)

else:
    st.info("Isi parameter di sidebar, lalu klik **Prediksi Sekarang**.")
