import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
# FUNGSI PREDIKSI (SINKRON DENGAN MODELING.PY)
# ============================================================
def predict_depth(year, lat, lon, mag, gap):
    DEFAULTS = {
        "dmin": df["dmin"].mean(),
        "rms": df["rms"].mean(),
        "herr": df["horizontalError"].mean(),
        "magerr": df["magError"].mean()
    }

    X = np.array([[year, lat, lon, mag, gap,
                   DEFAULTS["dmin"], DEFAULTS["rms"],
                   DEFAULTS["herr"], DEFAULTS["magerr"]]])

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
# INPUT SIDEBAR
# ============================================================
st.sidebar.title("🌍 Input Data Gempa")

year = st.sidebar.selectbox("Tahun", YEARS)
lat = st.sidebar.slider("Latitude", -12.0, 8.0, -2.0)
lon = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.slider("Magnitudo", 2.0, 9.0, 5.0)
gap = st.sidebar.slider("Gap", 0, 360, 100)
dmin = st.sidebar.slider("Dmin", 0.0, 20.0, 5.0)
rms = st.sidebar.slider("RMS", 0.0, 2.0, 0.5)
herr = st.sidebar.slider("Horizontal Error", 0.0, 30.0, 5.0)
magerr = st.sidebar.slider("Magnitude Error", 0.0, 0.5, 0.05)

btn = st.sidebar.button("🔍 Prediksi Sekarang")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Hasil Prediksi", "📊 Grafik", "📁 Info Dataset"])


# ============================================================
# TAB 1 — HASIL PREDIKSI
# ============================================================
with tab1:
    if btn:
        with st.spinner("⏳ Menganalisis data gempa..."):
            pred, proba = predict_depth(year, lat, lon, mag, gap)

        # ===================== HASIL PREDIKSI =====================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi Kedalaman Gempa")

        st.write(f"<span class='badge badge-{pred}'>Kelas {pred}</span>", unsafe_allow_html=True)
        st.write(f"**Interpretasi:** {CLASS_MAP[pred]}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ===================== RINGKASAN INPUT =====================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📝 Ringkasan Input Pengguna")

        st.write(f"""
        - **Tahun**: {year}  
        - **Latitude**: {lat}  
        - **Longitude**: {lon}  
        - **Magnitudo**: {mag}  
        - **Gap**: {gap}  
        - **Dmin**: {dmin}  
        - **RMS**: {rms}  
        - **Horizontal Error**: {herr}  
        - **Magnitude Error**: {magerr}  
        """)

        st.markdown("</div>", unsafe_allow_html=True)

        # ===================== PROBABILITAS =====================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Probabilitas Kelas")
        for i, p in enumerate(proba):
            st.write(f"- **Kelas {i}** ({CLASS_MAP[i]}): `{p*100:.2f}%`")
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
            marker=dict(
                color=["#e74c3c","#f1c40f","#27ae60"],
                line=dict(color="#0a3d3f", width=1.5)
            )
        ))
        fig.update_layout(
            title="Probabilitas Prediksi",
            template="plotly_white",
            plot_bgcolor="#f4fefb",
            paper_bgcolor="#f4fefb",
            yaxis_title="Probabilitas"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Grafik muncul setelah prediksi.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Grafik Kedalaman Rata-rata Per Tahun
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Depth Class Rata-rata per Tahun")

    df_year_avg = df.groupby("year")["depth"].mean().reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_year_avg["year"],
        y=df_year_avg["depth"],
        mode="lines+markers",
        marker=dict(size=8, color="#16a085"),
        line=dict(color="#1abc9c", width=3)
    ))

    fig2.update_layout(
        title="Trend Kedalaman Rata-rata per Tahun",
        template="plotly_white",
        plot_bgcolor="#f4fefb",
        paper_bgcolor="#f4fefb",
        xaxis_title="Tahun",
        yaxis_title="Kedalaman (km)"
    )

    st.plotly_chart(fig2, use_container_width=True)

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


    # Gambar: Distribusi Kelas
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🖼️ Distribusi Kelas Kedalaman")

    import seaborn as sns
    fig1, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df["depth_class"], palette="viridis", ax=ax)
    ax.set_title("Distribusi Depth Class")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Data")
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)


    # Gambar: Histogram Magnitudo
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🖼️ Histogram Magnitudo")

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.histplot(df["mag"], bins=30, kde=True, color="#1abc9c", ax=ax2)
    ax2.set_title("Distribusi Magnitudo Gempa")
    ax2.set_xlabel("Magnitudo")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)


    # Gambar: Scatter lokasi
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🖼️ Sebaran Lokasi Gempa")

    fig3, ax3 = plt.subplots(figsize=(6,5))
    ax3.scatter(df["longitude"], df["latitude"], s=8, alpha=0.5, color="#e67e22")
    ax3.set_title("Sebaran Lokasi Gempa")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)


    # Gambar: Depth per tahun
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🖼️ Rata-rata Kedalaman per Tahun")

    df_year_avg = df.groupby("year")["depth"].mean().reset_index()

    fig4, ax4 = plt.subplots(figsize=(6,4))
    ax4.plot(df_year_avg["year"], df_year_avg["depth"],
             marker="o", color="#16a085", linewidth=3)
    ax4.set_title("Rata-rata Kedalaman Gempa per Tahun")
    ax4.set_xlabel("Tahun")
    ax4.set_ylabel("Kedalaman (km)")
    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)
