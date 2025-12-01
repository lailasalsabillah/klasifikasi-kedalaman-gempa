import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Prediksi Kedalaman Gempa",
    page_icon="🔮",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("dataset_gempa.csv")

@st.cache_resource
def load_model():
    # sesuaikan path model
    model = joblib.load("models/xgb_depth_class.pkl")
    return model

df = load_data()
model = load_model()

st.title("🔮 Prediksi Klasifikasi Kedalaman Gempa")

st.markdown(
    """
Masukkan parameter gempa untuk memprediksi **kategori kedalaman**:  
**Shallow (<70 km), Intermediate (70–300 km), atau Deep (>300 km).**
"""
)

# Ambil range dari dataset untuk membantu input
mag_min, mag_max = float(df["magnitudo"].min()), float(df["magnitudo"].max())
lat_min, lat_max = float(df["lintang"].min()), float(df["lintang"].max())
lon_min, lon_max = float(df["bujur"].min()), float(df["bujur"].max())

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        magnitudo = st.slider(
            "Magnitudo",
            min_value=round(mag_min, 1),
            max_value=round(mag_max, 1),
            value=round(np.median(df["magnitudo"]), 1),
            step=0.1,
        )
        lintang = st.slider(
            "Lintang (°)",
            min_value=float(lat_min),
            max_value=float(lat_max),
            value=float(np.median(df["lintang"])),
            step=0.01,
        )

    with col2:
        bujur = st.slider(
            "Bujur (°)",
            min_value=float(lon_min),
            max_value=float(lon_max),
            value=float(np.median(df["bujur"])),
            step=0.01,
        )
        kedalaman_input = st.number_input(
            "Perkiraan Kedalaman (km) (opsional, jika ingin diisi)",
            min_value=0.0,
            value=float(max(1.0, df["kedalaman"].median())),
            step=1.0,
            help="Jika model kamu TIDAK memakai fitur kedalaman, bisa diabaikan di kode."
        )

    submitted = st.form_submit_button("Prediksi Klasifikasi Kedalaman")

if submitted:
    # ============
    # SESUAIKAN BAGIAN INI dengan fitur yang digunakan saat training model
    # Misal kalau saat training hanya pakai [magnitudo, lintang, bujur]:
    # features = np.array([[magnitudo, lintang, bujur]])
    #
    # Kalau pakai 4 fitur termasuk kedalaman:
    # features = np.array([[magnitudo, lintang, bujur, kedalaman_input]])
    # ============
    try:
        features = np.array([[magnitudo, lintang, bujur]])  # default: 3 fitur
        pred = model.predict(features)[0]

        # Jika model keluarkan angka 0/1/2 → mapping ke label
        mapping = {
            0: "Shallow (<70 km)",
            1: "Intermediate (70–300 km)",
            2: "Deep (>300 km)",
            "Shallow": "Shallow (<70 km)",
            "Intermediate": "Intermediate (70–300 km)",
            "Deep": "Deep (>300 km)",
        }

        label = mapping.get(pred, str(pred))

        st.success(f"📌 Hasil Prediksi Klasifikasi: **{label}**")

        with st.expander("Detail Input"):
            st.write(
                {
                    "magnitudo": magnitudo,
                    "lintang": lintang,
                    "bujur": bujur,
                    "kedalaman_input": kedalaman_input,
                }
            )

        st.markdown(
            """
**Interpretasi Umum:**

- **Shallow (<70 km)** → biasanya lebih terasa di permukaan, berpotensi merusak.
- **Intermediate (70–300 km)** → getaran dirasakan luas, tapi dampak di permukaan bisa beragam.
- **Deep (>300 km)** → biasanya getaran luas namun efek permukaan lebih kecil.
"""
        )
    except Exception as e:
        st.error("Terjadi error saat memanggil model. Cek kembali urutan fitur yang digunakan.")
        st.exception(e)
