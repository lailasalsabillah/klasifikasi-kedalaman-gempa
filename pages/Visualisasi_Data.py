import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Visualisasi Dataset Gempa")

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("dataset_gempa.csv")
        return df, None
    except Exception as e:
        return None, str(e)

df, err = load_dataset()

if err:
    st.error("Gagal memuat dataset.")
    st.code(err)
else:
    st.success("Dataset berhasil dimuat.")
    st.dataframe(df.head())

    st.subheader("Distribusi Magnitudo")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["mag"], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Scatter Plot: Magnitude vs Depth")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="mag", y="depth", ax=ax2)
    st.pyplot(fig2)
