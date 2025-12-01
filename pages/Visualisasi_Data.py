import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Visualisasi Dataset Gempa")

df = pd.read_csv("dataset_gempa.csv")

st.subheader("Distribusi Magnitude")
fig1, ax1 = plt.subplots()
ax1.hist(df["mag"], bins=30, color="orange")
st.pyplot(fig1)

st.subheader("Distribusi Kedalaman")
fig2, ax2 = plt.subplots()
ax2.hist(df["depth"], bins=30, color="blue")
st.pyplot(fig2)

st.subheader("Heatmap Korelasi")
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)
