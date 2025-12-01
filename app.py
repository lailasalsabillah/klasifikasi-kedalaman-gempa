import streamlit as st

st.set_page_config(
    page_title="Sistem Prediksi Kedalaman Gempa",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Sistem Prediksi Kedalaman Gempa Bumi")
st.write("""
Selamat datang di aplikasi **Prediksi Kedalaman Gempa** berbasis **LSTM & XGBoost**.

Gunakan menu di sebelah kiri untuk:
- 🔎 Melakukan prediksi
- 📊 Melihat visualisasi dataset
- 🗺️ Menampilkan peta lokasi gempa
- 📥 Mengunduh hasil prediksi
""")
