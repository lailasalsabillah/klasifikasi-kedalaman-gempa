import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Konfigurasi Direktori
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_depth_class.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset-gempa.csv")

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ scaler.pkl tidak ditemukan. Jalankan modeling.py dulu.")

if not os.path.exists(XGB_PATH):
    raise FileNotFoundError("❌ xgb_depth_class.pkl tidak ditemukan. Jalankan modeling.py dulu.")

scaler = joblib.load(SCALER_PATH)
xgb_model = joblib.load(XGB_PATH)

# -----------------------------
# Load Dataset (ambil tahun)
# -----------------------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("❌ dataset-gempa.csv tidak ditemukan.")

df_year = pd.read_csv(DATASET_PATH)

if "year" in df_year.columns:
    year_min = int(df_year["year"].min())
    year_max = int(df_year["year"].max())
    YEARS = list(range(year_min, year_max + 1))
else:
    YEARS = [2020, 2021, 2022, 2023, 2024]

# -----------------------------
# Mapping kelas
# -----------------------------
CLASS_MAP = {
    0: "Shallow (< 70 km) - Potensi kerusakan tinggi",
    1: "Intermediate (70–300 km) - Potensi kerusakan sedang",
    2: "Deep (> 300 km) - Relatif lebih aman",
}

# nilai default fitur lain
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

    proba = (
        xgb_model.predict_proba(X_scaled)[0]
        if hasattr(xgb_model, "predict_proba")
        else None
    )

    return int(y_pred), proba

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Prediksi Kedalaman Gempa", layout="wide")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.title("🌋 Input Parameter Gempa")

year = st.sidebar.selectbox("Tahun Kejadian", YEARS, index=len(YEARS)-1)
latitude = st.sidebar.slider("Latitude", -10.0, 10.0, -2.0, 0.01)
longitude = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0, 0.01)
mag = st.sidebar.slider("Magnitudo (Mw)", 2.0, 9.0, 5.0, 0.1)
gap = st.sidebar.slider("Gap", 0.0, 360.0, 100.0, 1.0)

predict_button = st.sidebar.button("🔍 Prediksi Sekarang")
st.sidebar.caption("Parameter lain seperti RMS, Dmin, Error diisi otomatis.")

# -----------------------------
# MAIN PAGE
# -----------------------------
st.title("🔎 Prediksi Kedalaman Gempa Bumi")
st.write("""
Aplikasi ini memprediksi **kelas kedalaman gempa bumi** menggunakan model XGBoost.
Isi parameter di sidebar kiri lalu klik *Prediksi Sekarang*.
""")
st.markdown("---")

# =========================================================
# ==================== HASIL PREDIKSI =====================
# =========================================================
if predict_button:
    y_pred, proba = predict_depth_class(year, latitude, longitude, mag, gap)

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** `{y_pred}`")
    st.write(f"**Interpretasi:** {CLASS_MAP[y_pred]}")

    # -------------------------
    # Probabilitas Teks
    # -------------------------
    if proba is not None:
        st.subheader("📊 Probabilitas Kelas")
        for i, p in enumerate(proba):
            st.write(f"- **Kelas {i}**: {p*100:.2f}%")

        # -------------------------
        # Grafik Bar Chart Probabilitas
        # -------------------------
        st.subheader("📉 Grafik Probabilitas")

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Kelas 0", "Kelas 1", "Kelas 2"]
        colors = ["#ff6b6b", "#feca57", "#1dd1a1"]

        ax.bar(labels, proba, color=colors)
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        ax.set_title("Probabilitas Model untuk Setiap Kelas")

        for i, p in enumerate(proba):
            ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center")

        st.pyplot(fig)

    # ======================================================
    # GRAFIK MAGNITUDO SEPANJANG TAHUN
    # ======================================================
    st.subheader("📈 Grafik Rata-rata Magnitudo per Tahun")

    if "year" in df_year.columns and "mag" in df_year.columns:
        df_mag_year = (
            df_year.groupby("year")["mag"]
            .mean()
            .reset_index()
            .sort_values("year")
        )

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(
            df_mag_year["year"],
            df_mag_year["mag"],
            marker="o",
            linestyle="-",
            linewidth=2,
            color="#3498db"
        )

        ax2.set_title("Rata-rata Magnitudo Gempa Tiap Tahun")
        ax2.set_xlabel("Tahun")
        ax2.set_ylabel("Magnitudo (Mw)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        for x, yv in zip(df_mag_year["year"], df_mag_year["mag"]):
            ax2.text(x, yv + 0.03, f"{yv:.2f}", ha="center")

        st.pyplot(fig2)

    else:
        st.warning("Dataset tidak memiliki kolom year atau mag.")

else:
    st.info("Silakan isi parameter di sidebar lalu klik **Prediksi Sekarang**.")
