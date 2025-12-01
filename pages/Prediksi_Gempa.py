import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import base64

st.title("🔎 Prediksi Kedalaman Gempa")

# Load model
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")
lstm_model = load_model("models/lstm_depth_class.keras")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

# Input user
with st.sidebar:
    st.header("Masukkan Parameter Gempa")
    lat = st.number_input("Latitude", -20.0, 20.0, 0.5)
    lon = st.number_input("Longitude", 80.0, 150.0, 110.0)
    mag = st.number_input("Magnitude", 3.0, 10.0, 5.0)
    gap = st.number_input("Gap", 0, 300, 40)
    dmin = st.number_input("Dmin", 0.0, 30.0, 2.0)
    rms = st.number_input("RMS", 0.0, 3.0, 0.7)
    herror = st.number_input("Horizontal Error", 0.0, 50.0, 8.0)
    derror = st.number_input("Depth Error", 0.0, 30.0, 6.0)
    magerr = st.number_input("Magnitude Error", 0.0, 1.0, 0.1)
    year = st.number_input("Year", 2000, 2030, 2023)

    btn = st.button("Prediksi Gempa")

if btn:
    data = np.array([[lat, lon, mag, gap, dmin, rms, herror, derror, magerr, year]])
    scaled = scaler.transform(data)

    # XGBoost
    xgb_pred = xgb_model.predict(scaled)[0]

    # LSTM
    lstm_pred = np.argmax(lstm_model.predict(scaled.reshape(1,1,10)), axis=1)[0]

    st.subheader("📌 Hasil Prediksi Machine Learning")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**XGBoost** memprediksi: **{label_map[xgb_pred]}**")

    with col2:
        st.warning(f"**LSTM** memprediksi: **{label_map[lstm_pred]}**")

    # Grafik magnitude
    fig, ax = plt.subplots()
    ax.bar(["Magnitude"], [mag], color="red")
    ax.set_title("Grafik Magnitude")
    st.pyplot(fig)

    # Tabel input
    df = pd.DataFrame(data, columns=[
        "Latitude","Longitude","Magnitude","Gap","Dmin",
        "RMS","HorizontalError","DepthError","MagError","Year"
    ])
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV Input", csv, "input_data.csv", "text/csv")
