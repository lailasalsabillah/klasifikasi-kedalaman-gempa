import os
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CUSTOM CSS 
# ============================================================
st.markdown("""
<style>

:root {
    --primary-color: #E63946;      /* Earthquake Red */
    --secondary-color: #457B9D;    /* Ash Blue */
    --background: #F8F9FA;         /* Light Gray Background */
    --card-bg: #FFFFFF;
    --card-border: #DDE5EC;
    --text-dark: #1D3557;          /* Deep Navy Text */
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
    color: var(--secondary-color);
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
# FUNGSI PREDIKSI
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
- 🚨 Peringatan Kedalaman  
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
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Hasil Prediksi", "📊 Grafik", "📁 Info Dataset"])

# ============================================================
# TAB 1 — HASIL PREDIKSI 
# ============================================================
with tab1:

    if not btn:
        st.info("👈 Masukkan input pada sidebar, lalu klik *Prediksi Sekarang*.")
        st.stop()

    # ===== JALANKAN MODEL =====
    pred, proba = predict_depth(year, lat, lon, depth, gap, dmin, rms, herr, magerr)

    # ============================================================
    # 1. RINGKASAN INPUT
    # ============================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Ringkasan Input Pengguna")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **Tahun:** {year}")
        st.write(f"- **Latitude:** {lat}")
        st.write(f"- **Longitude:** {lon}")
        st.write(f"- **Kedalaman:** {depth} km")
    with col2:
        st.write(f"- **Gap:** {gap}")
        st.write(f"- **Dmin:** {dmin}")
        st.write(f"- **RMS:** {rms}")
        st.write(f"- **Horizontal Error:** {herr}")
        st.write(f"- **Magnitude Error:** {magerr}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # 2. HASIL PREDIKSI
    # ============================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Hasil Prediksi Kedalaman Gempa")

    st.markdown(
        f"<span class='badge badge-{pred}' style='font-size:22px;'>Kelas {pred}</span>",
        unsafe_allow_html=True
    )
    st.write(f"**Interpretasi:** {CLASS_MAP[pred]}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # 3. PROBABILITAS PREDIKSI
    # ============================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Probabilitas Prediksi")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    for i, p in enumerate(proba):
        st.write(f"**Kelas {i}** — {CLASS_MAP[i]}: `{p*100:.2f}%`")

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # 4. KESIMPULAN (PARAGRAF DALAM KOTAK)
    # ============================================================
    max_proba = np.max(proba) * 100

    # ===== GENERATE PARAGRAF KESIMPULAN =====
    if pred == 0:
        explanation = f"""
        Gempa dangkal (< 70 km) memiliki potensi kerusakan yang tinggi karena pusat gempa berada dekat dengan permukaan bumi. 
        Getaran biasanya terasa lebih kuat dan dapat menyebabkan dampak signifikan pada bangunan dan lingkungan sekitar. 
        Model memprediksi kategori ini dengan tingkat keyakinan {max_proba:.2f}%.
        """
        box_color = "#FDECEA"
        border_color = "#E63946"

    elif pred == 1:
        explanation = f"""
        Gempa menengah (70–300 km) memiliki dampak yang sedang. Getaran biasanya masih terasa, tetapi tidak sekuat gempa dangkal. 
        Karena berada lebih dalam, energi gempa sebagian teredam sebelum mencapai permukaan. 
        Model memberikan prediksi ini dengan tingkat keyakinan {max_proba:.2f}%.
        """
        box_color = "#FFF9DB"
        border_color = "#F1C40F"

    else:
        explanation = f"""
        Gempa dalam (> 300 km) umumnya tidak menimbulkan kerusakan besar karena sumber gempa sangat jauh dari permukaan bumi. 
        Energi getaran sebagian besar teredam sebelum mencapai permukaan sehingga getarannya melemah. 
        Model mengkategorikan gempa ini dengan tingkat keyakinan {max_proba:.2f}%.
        """
        box_color = "#E8F8F5"
        border_color = "#2ECC71"

    # ===== TAMPILKAN KOTAK KESIMPULAN =====
    st.markdown(f"""
    <div style='
        background:{box_color};
        border-left:6px solid {border_color};
        padding:20px;
        border-radius:15px;
        margin-top:10px;
        margin-bottom:25px;
    '>
        <h4 style='margin-bottom:10px; color:{border_color};'>
            🧠 Kesimpulan
        </h4>
        <p style='font-size:16px; color:#1D3557; line-height:1.7; text-align:justify;'>
            {explanation}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — GRAFIK
# ============================================================
with tab2:

    # Grafik probabilitas
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Probabilitas Kelas (Plotly)")

    if btn:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Kelas 0", "Kelas 1", "Kelas 2"],
            y=proba,
            marker=dict(color=["#E63946", "#F1C40F", "#2ECC71"])
        ))
        fig.update_layout(title="Probabilitas Prediksi")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Grafik muncul setelah prediksi dilakukan.")

    st.markdown("</div>", unsafe_allow_html=True)

    # HISTOGRAM KEDALAMAN
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Histogram Kedalaman Gempa")

    fig_hist, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["depth"], bins=30, kde=True, color="#457B9D", ax=ax)
    st.pyplot(fig_hist)
    st.markdown("</div>", unsafe_allow_html=True)

    # SCATTER
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Scatter Plot: Kedalaman vs Magnitudo")

    fig_scatter, ax2 = plt.subplots(figsize=(6,4))
    ax2.scatter(df["depth"], df["mag"], alpha=0.5, color="#E63946")
    st.pyplot(fig_scatter)
    st.markdown("</div>", unsafe_allow_html=True)

    # BOXPLOT
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📦 Boxplot Kedalaman Gempa")

    fig_box, ax3 = plt.subplots(figsize=(5,4))
    sns.boxplot(df["depth"], color="#A8DADC", ax=ax3)
    st.pyplot(fig_box)
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

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Dataset (CSV)", csv, "dataset_gempa.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)
