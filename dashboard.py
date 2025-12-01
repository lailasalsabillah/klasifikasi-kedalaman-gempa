# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    layout="wide",
    page_icon="🌋"
)

# ============================
# SESSION STATE PAGE MANAGER
# ============================
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Home 🏠"

# Fungsi untuk mengubah menu saat tombol ditekan
def set_menu(menu_name):
    st.session_state.selected_menu = menu_name

# ============================
# LOAD MODEL
# ============================
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

try:
    lstm_model = load_model("models/lstm_depth_class.keras")
    lstm_ok = True
except:
    lstm_ok = False

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

danger_map = {
    0: ("Bahaya Tinggi", "red"),
    1: ("Bahaya Sedang", "orange"),
    2: ("Bahaya Rendah", "blue")
}

# ============================
# SIDEBAR
# ============================
st.sidebar.title("🌋 Prediksi Kedalaman Gempa")

# Tombol Sidebar untuk navigasi menu
if st.sidebar.button("Home 🏠", use_container_width=True):
    set_menu("Home 🏠")

if st.sidebar.button("Prediksi 📊", use_container_width=True):
    set_menu("Prediksi 📊")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Input Parameter Gempa (Slider Version)")

latitude = st.sidebar.slider("Latitude", -12.0, 10.0, -2.0)
longitude = st.sidebar.slider("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.slider("Magnitude", 3.0, 9.0, 4.5)
gap = st.sidebar.slider("Gap", 0, 300, 80)
dmin = st.sidebar.slider("Dmin", 0.0, 30.0, 2.1)
rms = st.sidebar.slider("RMS", 0.0, 3.0, 0.7)
horizontalError = st.sidebar.slider("Horizontal Error", 0.0, 30.0, 8.0)
depthError = st.sidebar.slider("Depth Error", 0.0, 20.0, 6.0)
magError = st.sidebar.slider("Magnitude Error", 0.0, 1.0, 0.12)
year = st.sidebar.slider("Year", 2020, 2024, 2023)

# Tombol untuk menampilkan prediksi (akan mengubah ke menu Prediksi)
if st.sidebar.button("➡ Tampilkan Prediksi", use_container_width=True):
    set_menu("Prediksi 📊")

# ============================
# FUNCTION PREDICT
# ============================
def predict_models():
    data = np.array([[
        latitude, longitude, mag, gap, dmin, rms,
        horizontalError, depthError, magError, year
    ]])

    data_scaled = scaler.transform(data)

    xgb_pred = xgb_model.predict(data_scaled)[0]

    if lstm_ok:
        lstm_input = data_scaled.reshape((1, 1, data_scaled.shape[1]))
        lstm_pred = np.argmax(lstm_model.predict(lstm_input), axis=1)[0]
    else:
        lstm_pred = None

    return data, xgb_pred, lstm_pred

# --- Tampilan Konten Berdasarkan Menu ---
st.title(st.session_state.selected_menu)

if st.session_state.selected_menu == "Home 🏠":
    st.markdown(
        """
        <style>
        .stApp{
            background-color: #d0eced; /* Warna latar belakang */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🌋 Prediksi Kedalaman Gempa Bumi")
    st.write("""
    Selamat datang di aplikasi prediksi kedalaman gempa bumi.

    Klik **➡ Tampilkan Prediksi** di sidebar untuk melakukan prediksi berdasarkan parameter yang kamu masukkan.
    """)

elif st.session_state.selected_menu == "Prediksi 📊":
    st.markdown(
        """
        <style>
        .stApp{
            background-color: #d0eced; /* Warna latar belakang */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("📊 Hasil Prediksi Kedalaman Gempa")

    data, xgb_pred, lstm_pred = predict_models()

    df_input = pd.DataFrame(data, columns=[
        "Latitude", "Longitude", "Magnitude", "Gap", "Dmin",
        "RMS", "HorizontalError", "DepthError", "MagError", "Year"
    ])

    # ============================
    # XGBOOST CARD
    # ============================
    xgb_label = label_map[xgb_pred]
    bahaya_xgb, color_xgb = danger_map[xgb_pred]

    st.subheader("📊 Prediksi XGBoost")
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 12px;
                    background-color:#f8f9fa;
                    border-left:12px solid {color_xgb};">

            <h3 style="color:{color_xgb};">{xgb_label}</h3>
            <p><i>Prediksi model XGBoost</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ============================
    # LSTM CARD
    # ============================
    if lstm_ok:
        lstm_label = label_map[lstm_pred]
        bahaya_lstm, color_lstm = danger_map[lstm_pred]

        st.subheader("🤖 Prediksi LSTM")
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 12px;
                        background-color:#f8f9fa;
                        border-left:12px solid {color_lstm};">
                <h3 style="color:{color_lstm};">{lstm_label}</h3>
                <p><i>Prediksi model LSTM</i></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠ Model LSTM tidak ditemukan!")

    st.markdown("---")

    # ============================
    # GRAPH
    # ============================
    st.subheader("📈 Grafik Magnitudo")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Magnitude"], [mag], color="purple")
    st.pyplot(fig)

    st.markdown("---")

    # ============================
    # TABLE INPUT
    # ============================
    st.subheader("🧾 Parameter Input")
    st.dataframe(df_input, use_container_width=True)

    st.markdown("---")
