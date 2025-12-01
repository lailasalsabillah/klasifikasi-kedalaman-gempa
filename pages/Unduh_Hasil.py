import streamlit as st
import pandas as pd

st.title("📥 Unduh Dataset atau Hasil Prediksi")

df = pd.read_csv("dataset-gempa.csv")

# Download dataset lengkap
st.download_button(
    "Download Dataset Gempa (CSV)",
    df.to_csv(index=False),
    "dataset_gempa.csv",
    "text/csv"
)

st.info("Hasil prediksi juga dapat diunduh dari halaman Prediksi Gempa.")
