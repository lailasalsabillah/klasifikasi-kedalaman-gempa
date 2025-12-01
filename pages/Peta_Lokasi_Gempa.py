import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.title("🗺️ Peta Lokasi Gempa")

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("dataset_gempa.csv")
        return df, None
    except Exception as e:
        return None, str(e)

df, err = load_dataset()

if df is None:
    st.error("Dataset tidak ditemukan.")
else:
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            popup=f"Mag: {row['mag']} | Depth: {row['depth']} km",
            color="red",
            fill=True
        ).add_to(m)

    st_folium(m, height=500, width=800)
