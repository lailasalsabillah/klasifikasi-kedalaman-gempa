import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

.stApp { background-color: var(--background); }

.card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: var(--radius);
    border: 1.5px solid var(--card-border);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.header-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    color: var(--text-dark);
}

.header-sub {
    text-align: center;
    color:#16a085;
    font-size: 16px;
    margin-bottom: 20px;
}

.divider {
    height: 4px;
    background: linear-gradient(90deg,var(--primary-color),transparent);
    border-radius: 20px;
    margin: 20px 0;
}

.stButton button {
    background-color: var(--primary-color);
    color:white;
    border-radius: var(--radius);
    padding:10px 18px;
    border:none;
    transition:0.2s;
}
.stButton button:hover {
    background-color: var(--secondary-color);
    transform:scale(1.03);
}

.badge {
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

# Load models safely with error message
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_depth_class.pkl"))
except Exception as e:
    st.error("❌ Gagal memuat model. File model kemungkinan rusak atau tidak kompatibel.")
    st.code(str(e))
    st.stop()

df = pd.read_csv(os.path.join(BASE_DIR, "dataset-gempa.csv"))
YEARS = sorted(df["year"].unique())

CLASS_MAP = {
    0: "Shallow (< 70 km) – Sangat Berpotensi",
    1: "Intermediate (70–300 km)",
    2: "Deep (> 300 km) – Relatif Aman"
}


# ============================================================
# FUNGSI PREDIKSI — MENGEMBALIKAN X & X_scaled (DEBUG MODE)
# ============================================================
def predict_depth(year, lat, lon, mag, gap):
    DEFAULTS = {
        "dmin": df["dmin"].mean(),
        "rms": df["rms"].mean(),
        "herr": df["horizontalError"].mean(),
        "magerr": df["magError"].mean()
    }

    # TOTAL 9 FITUR SESUAI MODELING.PY
    X = np.array([[year, lat, lon, mag, gap,
                   DEFAULTS["dmin"], DEFAULTS["rms"],
                   DEFAULTS["herr"], DEFAULTS["magerr"]]])

    # Scaling
    X_scaled = scaler.transform(X)

    # Prediksi
    pred = xgb_model.predict(X_scaled)[0]
    proba = xgb_model.predict_proba(X_scaled)[0]

    return pred, proba, X, X_scaled


# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='header-title'>🌋 Klasifikasi Kedalaman Gempa</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>XGBoost Earthquake Depth Classifier</div>", unsafe_allow_html=True)
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

btn = st.sidebar.button("🔍 Prediksi Sekarang")


# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Hasil Prediksi", "📊 Grafik", "📁 Info Dataset"])


# ============================================================
# TAB 1 — HASIL PREDIKSI + DEBUG MODE
# ============================================================
with tab1:
    if btn:
        with st.spinner("⏳ Memproses prediksi..."):
            pred, proba, X_raw, X_scaled = predict_depth(year, lat, lon, mag, gap)

        # ==== DEBUG MODE ====
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🛠 DEBUG MODE — Data yang Masuk ke Model")

        st.write("### 🔹 Input Features (X RAW):")
        st.code(str(X_raw))

        st.write("### 🔹 Input Setelah Scaling (X_scaled):")
        st.code(str(X_scaled))

        st.write("### 🔹 Probabilitas Model:")
        st.write({
            "Kelas 0": float(proba[0]),
            "Kelas 1": float(proba[1]),
            "Kelas 2": float(proba[2]),
        })

        # Jika model condong 100% ke kelas 0 → model lama / rusak
        if proba[0] > 0.95 and proba[1] < 0.04 and proba[2] < 0.01:
            st.error("⚠ Model mendeteksi kelas 0 hampir 100% untuk semua input.")
            st.info("""
Masalah ini terjadi jika:
1. Model lama masih digunakan
2. Model belum di-train ulang (tanpa fitur depth)
3. File model corrupt / tidak kompatibel
4. Streamlit memuat model default fallback

SOLUSI:
- Jalankan ulang modeling.py
- Upload ulang seluruh isi folder /models
- Restart Streamlit Cloud (Manage App → Restart)
            """)

        st.markdown("</div>", unsafe_allow_html=True)

        # ==== HASIL PREDIKSI ====
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi Kedalaman Gempa")

        st.write(f"<span class='badge badge-{pred}'>Kelas {pred}</span>", unsafe_allow_html=True)
        st.write(f"**Interpretasi:** {CLASS_MAP[pred]}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ==== PROBABILITAS ====
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Probabilitas Kelas")

        for i, p in enumerate(proba):
            st.write(f"- **Kelas {i}** ({CLASS_MAP[i]}): `{p*100:.2f}%`")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("👈 Masukkan data di kiri lalu klik *Prediksi Sekarang*.")


# ============================================================
# TAB 2 — GRAFIK
# ============================================================
with tab2:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Probabilitas Kelas")

    if btn:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Kelas 0", "Kelas 1", "Kelas 2"],
            y=proba,
            marker=dict(color=["#e74c3c","#f1c40f","#27ae60"])
        ))
        fig.update_layout(
            title="Probabilitas Prediksi",
            template="plotly_white",
            plot_bgcolor="#f4fefb"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Prediksi diperlukan untuk menampilkan grafik ini.")

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
