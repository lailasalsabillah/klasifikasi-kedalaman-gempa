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
# LOAD MODEL
# ============================
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

# LOAD LSTM MODEL
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
# SIDEBAR INPUT
# ============================
st.sidebar.title("🔍 Input Parameter Gempa")

latitude = st.sidebar.number_input("Latitude", -12.0, 10.0, -2.0)
longitude = st.sidebar.number_input("Longitude", 90.0, 150.0, 120.0)
mag = st.sidebar.number_input("Magnitude", 3.0, 9.0, 4.5)
gap = st.sidebar.number_input("Gap", 0, 300, 80)
dmin = st.sidebar.number_input("Dmin", 0.0, 30.0, 2.1)
rms = st.sidebar.number_input("RMS", 0.0, 3.0, 0.7)
horizontalError = st.sidebar.number_input("Horizontal Error", 0.0, 30.0, 8.0)
depthError = st.sidebar.number_input("Depth Error", 0.0, 20.0, 6.0)
magError = st.sidebar.number_input("Magnitude Error", 0.0, 1.0, 0.12)
year = st.sidebar.number_input("Year", 2020, 2024, 2023)

predict_btn = st.sidebar.button("🔎 Prediksi Kedalaman Gempa")

# ============================
# FUNCTION PREDICT
# ============================
def predict_models():
    data = np.array([[
        latitude, longitude, mag, gap, dmin, rms,
        horizontalError, depthError, magError, year
    ]])

    # scaling
    data_scaled = scaler.transform(data)

    # XGBOOST predict
    xgb_pred = xgb_model.predict(data_scaled)[0]

    # LSTM predict = reshape (samples, timesteps, features)
    if lstm_ok:
        lstm_input = data_scaled.reshape((1,1, data_scaled.shape[1]))
        lstm_pred = np.argmax(lstm_model.predict(lstm_input), axis=1)[0]
    else:
        lstm_pred = None

    return data, xgb_pred, lstm_pred

# ============================
# TITLE
# ============================
st.title("🌋 Prediksi Kedalaman Gempa Bumi")
st.write("Model prediksi kedalaman berdasarkan parameter seismik menggunakan algoritma **XGBoost & LSTM**.")

st.markdown("---")

# ============================
# RESULT PAGE
# ============================
if predict_btn:

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

    st.subheader("📊 Hasil Prediksi XGBoost")
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 12px;
                    background-color:#f8f9fa;
                    border-left:12px solid {color_xgb};">

            <h3 style="color:{color_xgb};">{xgb_label}</h3>
            <p><i>Prediksi berdasarkan model XGBoost</i></p>
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

        st.subheader("🤖 Hasil Prediksi LSTM")
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 12px;
                        background-color:#f8f9fa;
                        border-left:12px solid {color_lstm};">

                <h3 style="color:{color_lstm};">{lstm_label}</h3>
                <p><i>Prediksi berdasarkan model LSTM</i></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠ Model LSTM belum tersedia dalam folder /models")

    st.markdown("---")

    # ============================
    # GRAPHIC VISUALIZATION
    # ============================
    st.subheader("📈 Visualisasi Magnitude Gempa")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Magnitude"], [mag], color="purple")
    ax.set_title("Magnitude dari Gempa")
    st.pyplot(fig)

    st.markdown("---")

    # ============================
    # SHOW TABLE
    # ============================
    st.subheader("🧾 Tabel Parameter Gempa")
    st.dataframe(df_input, use_container_width=True)

    st.markdown("---")

    # ============================
    # EXPLANATION
    # ============================
    st.subheader("ℹ Penjelasan Kategori Kedalaman Gempa")
    st.write("""
    **Shallow (<70 km)** → Energi belum meredam, sangat berbahaya.  
    **Intermediate (70–300 km)** → Bahaya sedang, jangkauan luas.  
    **Deep (>300 km)** → Energi meredam banyak, bahaya rendah.
    """)

else:
    st.info("Masukkan parameter di sidebar, lalu klik **Prediksi Kedalaman Gempa**.")
