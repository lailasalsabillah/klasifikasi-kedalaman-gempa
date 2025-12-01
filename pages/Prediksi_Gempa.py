import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# TITLE
# -------------------------------
st.title("🌍 Prediksi Kategori Kedalaman Gempa")
st.write("Model menggunakan XGBoost dan LSTM Neural Network")


# -------------------------------
# LOAD MODEL
# -------------------------------
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")
lstm_model = load_model("models/lstm_depth_class.keras")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

# -------------------------------
# FUNGSI PREDIKSI
# -------------------------------
def predict_depth(model_choice, df):
    scaled = scaler.transform(df)

    if model_choice == "XGBoost":
        pred = xgb_model.predict(scaled)[0]
    else:
        # LSTM membutuhkan dimensi tambahan [samples, timesteps, features]
        reshaped = np.expand_dims(scaled, axis=1)
        pred = np.argmax(lstm_model.predict(reshaped), axis=1)[0]

    return label_map[pred]


# -------------------------------
# INPUT MODE 1 – PREDIKSI DATA INDIVIDU
# -------------------------------
st.header("🔎 Prediksi Kedalaman Gempa (Input Satu Data)")

model_choice = st.selectbox("Pilih Model:", ["XGBoost", "LSTM"])

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", -20.0, 20.0)
    longitude = st.number_input("Longitude", 80.0, 150.0)
    mag = st.number_input("Magnitude", 3.0, 10.0)
    gap = st.number_input("Gap", 0, 300)
    dmin = st.number_input("Dmin", 0.0, 30.0)

with col2:
    rms = st.number_input("RMS", 0.0, 3.0)
    herror = st.number_input("Horizontal Error", 0.0, 50.0)
    derror = st.number_input("Depth Error", 0.0, 30.0)
    magerr = st.number_input("Magnitude Error", 0.0, 1.0)
    year = st.number_input("Year", 2000, 2030)


# Buat dataframe input
input_df = pd.DataFrame([[
    latitude, longitude, mag, gap, dmin,
    rms, herror, derror, magerr, year
]], columns=[
    "latitude", "longitude", "mag", "gap", "dmin",
    "rms", "horizontal_error", "depth_error", "mag_error", "year"
])

st.write("📘 **Data Input Anda:**")
st.dataframe(input_df)

if st.button("Prediksi Kedalaman Gempa"):
    result = predict_depth(model_choice, input_df)
    st.success(f"📌 **Hasil Prediksi: {result}**")


# -------------------------------
# INPUT MODE 2 – RENTANG PARAMETER (SIDEBAR)
# -------------------------------
with st.sidebar:
    st.header("📊 Prediksi Dengan Rentang Parameter")

    lat_range = st.slider("Latitude", -20.0, 20.0, (-10.0, 10.0))
    lon_range = st.slider("Longitude", 80.0, 150.0, (100.0, 120.0))
    mag_range = st.slider("Magnitude", 3.0, 10.0, (4.0, 6.0))
    gap_range = st.slider("Gap", 0, 300, (20, 80))
    dmin_range = st.slider("Dmin", 0.0, 30.0, (1.0, 5.0))
    rms_range = st.slider("RMS", 0.0, 3.0, (0.5, 1.5))
    herror_range = st.slider("Horizontal Error", 0.0, 50.0, (5.0, 10.0))
    derror_range = st.slider("Depth Error", 0.0, 30.0, (3.0, 8.0))
    magerr_range = st.slider("Magnitude Error", 0.0, 1.0, (0.05, 0.2))
    year_range = st.slider("Year", 2000, 2030, (2015, 2025))

    if st.button("Prediksi Dari Rentang"):
        # Ambil nilai rata-rata dari range
        data_avg = pd.DataFrame([[
            np.mean(lat_range), np.mean(lon_range), np.mean(mag_range),
            np.mean(gap_range), np.mean(dmin_range), np.mean(rms_range),
            np.mean(herror_range), np.mean(derror_range),
            np.mean(magerr_range), np.mean(year_range)
        ]], columns=[
            "latitude", "longitude", "mag", "gap", "dmin",
            "rms", "horizontal_error", "depth_error", "mag_error", "year"
        ])

        depth_res = predict_depth("XGBoost", data_avg)

        st.write("📘 **Data Rata-Rata Rentang:**")
        st.dataframe(data_avg)

        st.success(f"📌 **Hasil Prediksi Rentang: {depth_res}**")
