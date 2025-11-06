import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="World Development Clustering", layout="wide")

MODEL_DIR = "clustering_artifacts"

def safe_load(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

scaler = safe_load("scaler.joblib")
model = safe_load("chosen_model.joblib")
pca2 = safe_load("pca2.joblib")

if scaler is None or model is None:
    st.error("❌ Model artifacts missing. Please upload 'clustering_artifacts'.")
    st.stop()

st.title("🌍 World Development Clustering Dashboard")
st.write("Upload your dataset (same numeric columns used during training).")

uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=[np.number])

    expected_cols = scaler.mean_.shape[0]
    if numeric_df.shape[1] != expected_cols:
        st.error(f"❌ Expected {expected_cols} numeric columns, but got {numeric_df.shape[1]}.")
        st.stop()

    X_scaled = scaler.transform(numeric_df)
    labels = model.predict(X_scaled)

    df["Cluster"] = labels
    st.success("✅ Clustering Successful!")
    st.dataframe(df.head())

    st.subheader("📊 Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())

    if pca2:
        pcs = pca2.transform(X_scaled)
        pcs_df = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "Cluster": labels})

        st.subheader("🌐 PCA Visualization")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=pcs_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=80, ax=ax)
        st.pyplot(fig)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Clustered CSV", csv, "clustered_output.csv", "text/csv")

else:
    st.info("📌 Please upload a CSV file to begin.")
