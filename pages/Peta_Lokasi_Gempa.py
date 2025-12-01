import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.title("🗺️ Peta Lokasi Gempa")

df = pd.read_csv("dataset_gempa.csv")

m = folium.Map(location=[-2, 120], zoom_start=5)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        popup=f"Mag: {row['mag']} | Depth: {row['depth']} km",
        color="red",
        fill=True
    ).add_to(m)

st_folium(m, width=800)
