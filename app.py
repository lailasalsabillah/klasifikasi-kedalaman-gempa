import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    color: #198f84;
    margin-bottom: 25px;
}

/* Styled Divider */
.divider {
    height: 3px;
    background: linear-gradient(90deg, var(--primary-color), transparent);
    margin: 15px 0;
    border-radius: 20px;
}

/* Buttons */
.stButton button {
    background-color: var(--primary-color);
    color: white;
    border-radius: var(--radius);
    padding: 10px 16px;
    border: none;
    transition: 0.2s;
}

.stButton button:hover {
    background-color: var(--secondary-color);
    transform: scale(1.03);
}

/* Result Badge */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 14px;
    color: white;
    font-weight: 600;
}

.badge-0 { background: #e74c3c; }
.badge-1 { background: #f1c40f; }
.badge-2 { background: #27ae60; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL + DATASET
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_depth_class.pkl"))

df_year = pd.read_csv(os.path.join(BASE_DIR, "dataset-gempa.csv"))
YEARS = sorted(df_year["year"].unique())

CLASS_MAP = {
    0: "Shallow (< 70 km) – Sangat Berpotensi",
    1: "Intermediate (70–300 km)",
    2: "Deep (> 300 km) – Relatif Aman"
}


# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_depth(year, lat, lon, mag, gap):
    DEFAULTS = {
        "dmin": df_year["dmin"].mean(),
        "rms": df_year["rms"].mean(),
        "herr": df_year["horizontalError"].mean(),
        "derr": df_year["depthError"].mean(),
        "magerr": df_year["magError"].mean()
    }

    X = np.array([[year, lat, lon, mag, gap,
                   DEFAULTS["dmin"], DEFAULTS["rms"],
                   DEFAULTS["herr"], DEFAULTS["derr"],
                   DEFAULTS["magerr"]]])

    X_scaled = scaler.transform(X)
    pred = xgb_model.predict(X_scaled)[0]
    proba = xgb_model.predict_proba(X_scaled)[0]

    return pred, proba


# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='header-title'>🌋 Prediksi Kedalaman Gempa Tektonik</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>Machine Learning – XGBoost Earthquake Depth Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR INPUT
# ============================================================
st.sidebar.title("🌍 Input Data Gempa")

year = st.sidebar.selectbox("Tahun", YEARS)
lat = st.sidebar.slider("Latitude", -12.0, 8.0, -2.0)
lon = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.slider("Magnitudo", 2.0, 9.0, 5.0)
gap = st.sidebar.slider("Gap", 0, 360, 100)

btn_predict = st.sidebar.button("🔍 Prediksi Sekarang")


# ============================================================
# MAIN OUTPUT AREA
# ============================================================
if btn_predict:

    with st.spinner("⏳ Sedang menganalisis data gempa..."):
        pred, proba = predict_depth(year, lat, lon, mag, gap)

    # ===================== CARD HASIL =====================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📌 Hasil Prediksi")

    st.write(f"""
    <span class='badge badge-{pred}'>
        Kelas {pred}
    </span>
    """, unsafe_allow_html=True)

    st.write(f"**Interpretasi:** {CLASS_MAP[pred]}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ===================== PROBABILITAS =====================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Probabilitas Kelas")

    for i, p in enumerate(proba):
        st.write(f"- **Kelas {i}** ({CLASS_MAP[i]}): `{p*100:.2f}%`")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Masukkan parameter gempa di sidebar, lalu klik **Prediksi Sekarang**.")
