import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, timedelta
import pytz

# ================================
# KONFIGURASI PAGE UTAMA
# ================================
st.set_page_config(
    page_title="Deteksi Gempa Indonesia",
    page_icon="🌍",
    layout="wide"
)

# Header Branding aplikasi (Tema Light Blue)
st.markdown(
    """
    <h1 style='text-align: center; color:#0b5394;'>
        🌍 Deteksi Gempa Bumi Indonesia
    </h1>
    <p style='text-align: center; font-size:18px; color:#444;'>
        Monitoring gempa bumi real-time + Analisis & Prediksi Kedalaman Gempa
    </p>
    """,
    unsafe_allow_html=True
)

# ================================
# FUNGSI AMBIL DATA GEMPA USGS
# ================================
def fetch_usgs_indonesia_earthquakes():
    """Mengambil data gempa dari USGS dengan filter area Indonesia"""
    try:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'endtime': datetime.now().strftime('%Y-%m-%d'),
            'minlatitude': -11,
            'maxlatitude': 6,
            'minlongitude': 95,
            'maxlongitude': 141,
            'minmagnitude': 2.5,
            'limit': 100
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        earthquakes = []
        
        for feature in data['features']:
            try:
                properties = feature['properties']
                geometry = feature['geometry']
                
                magnitude = properties.get('mag', 0)
                place = properties.get('place', 'Unknown location')
                time_ms = properties.get('time', 0)
                depth = geometry['coordinates'][2] if len(geometry['coordinates']) > 2 else 0
                
                utc_time = pd.to_datetime(time_ms, unit='ms')
                local_time = utc_time.tz_localize('UTC').tz_convert(pytz.timezone('Asia/Jakarta'))

                earthquakes.append({
                    "tanggal": local_time.strftime("%d-%b-%Y"),
                    "jam": local_time.strftime("%H:%M:%S"),
                    "lintang": geometry['coordinates'][1],
                    "bujur": geometry['coordinates'][0],
                    "magnitudo": magnitude,
                    "kedalaman": int(depth),
                    "wilayah": place,
                    "potensi_tsunami": "Tidak berpotensi tsunami" if magnitude < 7.0 else "Berpotensi tsunami",
                    "waktu_kejadian": local_time
                })
                
            except:
                continue
        
        return pd.DataFrame(earthquakes)
        
    except Exception:
        st.error("Error mengambil data dari USGS. Menampilkan data contoh.")
        return create_dummy_data()

# ================================
# DATA DUMMY (Backup)
# ================================
def create_dummy_data():
    return pd.DataFrame([
        {"tanggal": "16-Jun-2025", "jam": "10:30:00", "lintang": -6.2, "bujur": 106.8,
         "magnitudo": 4.2, "kedalaman": 15, "wilayah": "Jakarta Selatan",
         "potensi_tsunami": "Tidak berpotensi tsunami",
         "waktu_kejadian": datetime.now(pytz.timezone('Asia/Jakarta'))},
    ])

# ================================
# AMBIL DATA
# ================================
with st.spinner("📡 Mengambil data gempa..."):
    earthquake_data = fetch_usgs_indonesia_earthquakes()

st.markdown("---")

# ================================
# STATISTIK RINGKAS
# ================================
if not earthquake_data.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Gempa", len(earthquake_data))

    with col2:
        st.metric("Magnitudo Tertinggi", f"{earthquake_data['magnitudo'].max():.1f}")

    with col3:
        st.metric("Kedalaman Rata-rata", f"{earthquake_data['kedalaman'].mean():.1f} km")

    with col4:
        st.metric("Gempa M ≥ 4.0", len(earthquake_data[earthquake_data['magnitudo'] >= 4]))

# ================================
# FILTER MAGNITUDO
# ================================
st.subheader("Filter Data")
min_magnitude = st.slider(
    "Magnitudo Minimum",
    min_value=float(earthquake_data['magnitudo'].min()),
    max_value=float(earthquake_data['magnitudo'].max()),
    value=2.5,
    step=0.1
)

filtered_data = earthquake_data[earthquake_data["magnitudo"] >= min_magnitude]

# ================================
# PETA GEMPA
# ================================
st.subheader("🗺️ Peta Gempa Bumi Indonesia (7 Hari Terakhir)")

if not filtered_data.empty:
    fig = px.scatter_mapbox(
        filtered_data,
        lat="lintang",
        lon="bujur",
        size="magnitudo",
        color="magnitudo",
        hover_name="wilayah",
        hover_data=["tanggal", "jam", "magnitudo", "kedalaman"],
        zoom=4,
        center={"lat": -2.5, "lon": 118},
        height=600,
        color_continuous_scale="Reds"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Tidak ada gempa dengan magnitudo minimum tersebut.")

# ================================
# TABEL DATA
# ================================
st.subheader("📋 Data Gempa Terkini")

display_data = filtered_data.sort_values("waktu_kejadian", ascending=False)
st.dataframe(
    display_data[['tanggal', 'jam', 'wilayah', 'magnitudo', 'kedalaman', 'potensi_tsunami']],
    use_container_width=True,
    height=350
)

csv = display_data.to_csv(index=False)
st.download_button("📥 Download Data CSV", csv, "gempa_indonesia.csv", "text/csv")

st.markdown("---")
st.caption("Data oleh USGS — Aplikasi oleh Sistem Prediksi Kedalaman Gempa")

# ================================
# SIDEBAR NAVIGASI
# ================================
st.sidebar.title("📌 Navigasi")
st.sidebar.page_link("app.py", label="🌍 Deteksi Gempa Indonesia")
st.sidebar.page_link("pages/1_Prediksi_Kedalaman.py", label="🔮 Prediksi Kedalaman Gempa")
st.sidebar.page_link("pages/2_Analisis_Dataset.py", label="📊 Analisis Dataset")
st.sidebar.markdown("---")
st.sidebar.info("Data real-time USGS area Indonesia.")

