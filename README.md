# Earthquake Depth Classification (LSTM & XGBoost)

Project ini melakukan klasifikasi kedalaman gempa bumi menjadi:
- Shallow (<70 km)
- Intermediate (70–300 km)
- Deep (>300 km)

Menggunakan model:
- LSTM (TensorFlow)
- XGBoost

Terdapat 2 bagian utama dalam project:
1. modeling.py → proses preprocessing, training, dan penyimpanan model (.pkl / .keras)
2. dashboard.py → aplikasi Streamlit yang memanggil model dan membuat prediksi

## Struktur Project
