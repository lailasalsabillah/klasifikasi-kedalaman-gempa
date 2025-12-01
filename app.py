import streamlit as st

st.set_page_config(page_title="Prediksi Kedalaman Gempa", layout="wide")

st.title("🌋 Prediksi Kedalaman Gempa Bumi")
st.write("""
Aplikasi ini memprediksi kategori kedalaman gempa (Shallow, Intermediate, Deep)
berdasarkan model **XGBoost** dan **LSTM**.
Gunakan menu di sidebar untuk membuka halaman lain.
""")

st.info("Pilih halaman *Prediksi Gempa*, *Visualisasi Data*, *Peta Lokasi*, atau *Unduh Hasil* melalui sidebar.")
