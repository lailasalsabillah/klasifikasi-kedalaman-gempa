import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Analisis Dataset Gempa",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("dataset_gempa.csv")

df = load_data()

st.title("📈 Analisis Dataset Gempa")

st.subheader("Preview Data")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

with col2:
    st.subheader("Jumlah Data per Klasifikasi")
    klas_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("klasifikasi:N", title="Klasifikasi"),
            y=alt.Y("count():Q", title="Jumlah"),
            tooltip=["klasifikasi", "count()"]
        )
    )
    st.altair_chart(klas_chart, use_container_width=True)

st.markdown("---")

# Histogram kedalaman
st.subheader("Distribusi Kedalaman (km)")
ked_chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("kedalaman:Q", bin=alt.Bin(maxbins=30), title="Kedalaman (km)"),
        y=alt.Y("count():Q", title="Jumlah"),
        tooltip=["count()"]
    )
)
st.altair_chart(ked_chart, use_container_width=True)

st.markdown("---")

# Time series per tanggal (jika bisa dikonversi)
if "tanggal" in df.columns:
    df_time = df.copy()
    df_time["tanggal"] = pd.to_datetime(df_time["tanggal"], errors="coerce")
    df_time = df_time.dropna(subset=["tanggal"])

    st.subheader("Jumlah Gempa per Tanggal")

    time_chart = (
        alt.Chart(df_time)
        .mark_line(point=True)
        .encode(
            x=alt.X("tanggal:T", title="Tanggal"),
            y=alt.Y("count():Q", title="Jumlah Gempa"),
            tooltip=["tanggal", "count()"]
        )
    )

    st.altair_chart(time_chart, use_container_width=True)
else:
    st.info("Kolom tanggal tidak ditemukan, grafik time series tidak dapat ditampilkan.")
