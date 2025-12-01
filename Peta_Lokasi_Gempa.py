import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.title("🗺️ Peta Lokasi Gempa")

df = pd.read_csv("dataset/earthquake_dataset.csv")

m = folium.Map(location=[-2, 118], zoom_start=5)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        popup=f"M {row['mag']} | Depth {row['depth']} km",
        color="red"
    ).add_to(m)

st_folium(m, width=800, height=480)
