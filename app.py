import streamlit as st

st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    page_icon="🌋",
    layout="wide"
)

# ============================
# CUSTOM BUTTON CSS
# ============================
st.markdown("""
<style>
.custom-button {
    display: inline-block;
    padding: 10px 18px;
    background-color: #4285F4;
    color: white !important;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 600;
    text-align: center;
    border: 1px solid #3367D6;
    transition: 0.2s;
}
.custom-button:hover {
    background-color: #3367D6;
}
</style>
""", unsafe_allow_html=True)


# ======================================
# SIDEBAR INPUT → versi elegan seperti Mount Jawa
# ======================================
st.sidebar.markdown("## 🧮 Input Parameter Gempa")

latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 0.0)
longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 0.0)
mag = st.sidebar.number_input("Magnitude", 0.0, 10.0, 5.0)
gap = st.sidebar.number_input("Gap", 0, 360, 80)
dmin = st.sidebar.number_input("Dmin", 0.0, 30.0, 2.0)
rms = st.sidebar.number_input("RMS", 0.0, 10.0, 0.7)
horizontalError = st.sidebar.number_input("Horizontal Error", 0.0, 50.0, 8.0)
depthError = st.sidebar.number_input("Depth Error", 0.0, 30.0, 5.0)
magError = st.sidebar.number_input("Magnitude Error", 0.0, 1.0, 0.1)
year = st.sidebar.number_input("Tahun", 2000, 2100, 2023)

# Tombol modern seperti Mount Jawa
predict_button = st.sidebar.button("🔍 Prediksi Sekarang")

# Bendera untuk berpindah halaman (session_state)
if "go_predict" not in st.session_state:
    st.session_state.go_predict = False

if predict_button:
    st.session_state.go_predict = True
    st.switch_page("pages/1_🔍_Prediksi_Gempa.py")


# ==========================================
# HALAMAN UTAMA
# ==========================================
st.title("🌋 Sistem Prediksi Kedalaman Gempa Bumi")

st.write("""
Selamat datang di aplikasi **Prediksi Kedalaman Gempa Bumi**.

Aplikasi ini memprediksi kategori kedalaman:
- **Shallow (<70 km)**
- **Intermediate (70–300 km)**
- **Deep (>300 km)**

Menggunakan model:
- **XGBoost**
- **LSTM**

Gunakan sidebar di kiri untuk memasukkan parameter gempa kemudian klik **Prediksi Sekarang**.
""")
