import streamlit as st

st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Sistem Prediksi Kedalaman Gempa Bumi")
st.write("""
Selamat datang di aplikasi **Prediksi Kedalaman Gempa Bumi** yang dibangun menggunakan model  
**LSTM** dan **XGBoost**.

Gunakan menu Sidebar di kiri (📑 Pages) untuk:
- ⚡ Melakukan Prediksi Kedalaman Gempa  
- 📊 Melihat Visualisasi Data  
- 🗺️ Melihat Peta Lokasi Gempa  
- 📥 Mengunduh Hasil Prediksi  

Aplikasi ini memanfaatkan dataset gempa 2020–2024 dan memprediksi kategori kedalaman:  
- Shallow (<70 km)
- Intermediate (70–300 km)
- Deep (>300 km)
