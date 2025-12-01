import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Klasifikasi Kedalaman Gempa",
    page_icon="🌊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_gempa.csv")
    # pastikan nama kolom sesuai
    return df

df = load_data()

st.title("📊 Dashboard Klasifikasi Kedalaman Gempa")
st.markdown(
    """
Aplikasi ini menampilkan **visualisasi data gempa bumi** dan  
**klasifikasi kedalaman gempa**: Shallow, Intermediate, dan Deep.
"""
)

# ──────────────────────────────
# Kartu Ringkasan
# ──────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### Total Data")
    st.metric(label="Jumlah gempa", value=len(df))

with col2:
    st.markdown("### Kedalaman Rata-rata")
    kedalaman_mean = df["kedalaman"].mean()
    st.metric(label="Rata-rata (km)", value=f"{kedalaman_mean:.1f}")

with col3:
    st.markdown("### Magnitudo Rata-rata")
    mag_mean = df["magnitudo"].mean()
    st.metric(label="Rata-rata M", value=f"{mag_mean:.2f}")

with col4:
    st.markdown("### Kelas Kedalaman")
    kelas_counts = df["klasifikasi"].value_counts()
    kelas_str = ", ".join([f"{k}: {v}" for k, v in kelas_counts.items()])
    st.write(kelas_str)

st.markdown("---")

# ──────────────────────────────
# Grafik distribusi klasifikasi
# ──────────────────────────────
st.subheader("Distribusi Klasifikasi Kedalaman")

chart_kelas = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("klasifikasi:N", title="Klasifikasi Kedalaman"),
        y=alt.Y("count():Q", title="Jumlah"),
        tooltip=["klasifikasi", "count()"]
    )
)

st.altair_chart(chart_kelas, use_container_width=True)

st.markdown("---")

# ──────────────────────────────
# Peta lokasi gempa (jika ada)
# ──────────────────────────────
if {"lintang", "bujur"}.issubset(df.columns):
    st.subheader("Peta Lokasi Gempa")

    df_map = df.rename(columns={"lintang": "lat", "bujur": "lon"})
    st.map(df_map[["lat", "lon"]].dropna())
else:
    st.info("Kolom lintang & bujur tidak ditemukan, peta tidak dapat ditampilkan.")

st.markdown("---")

# ──────────────────────────────
# Grafik Magnitudo vs Kedalaman
# ──────────────────────────────
st.subheader("Magnitudo vs Kedalaman")

scatter = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x=alt.X("magnitudo:Q", title="Magnitudo"),
        y=alt.Y("kedalaman:Q", title="Kedalaman (km)"),
        color=alt.Color("klasifikasi:N", title="Klasifikasi"),
        tooltip=["tanggal", "jam", "wilayah", "magnitudo", "kedalaman", "klasifikasi"]
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

st.markdown(
    """
🔎 **Tips**:  
- Gunakan menu di kiri (sidebar) untuk berpindah ke halaman **Prediksi Kedalaman** atau **Analisis Dataset**.
"""
)
