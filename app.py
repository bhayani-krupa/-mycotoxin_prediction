import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Pretrained Models
xgb_model = joblib.load("xgboost.pkl")  # Load trained XGBoost model
pca_model = joblib.load("pca_model.pkl")  # Load trained PCA model
scaler_model = joblib.load("scaler.pkl")  # Load trained Scaler model
trained_feature_names = joblib.load("feature_names.pkl")  # Load stored feature names from training

# Streamlit App UI
st.title("Spectral Data Prediction Using XGBoost 🚀")
st.write("Upload spectral data (CSV) to predict DON Concentration using XGBoost.")

# File Uploader
uploaded_file = st.file_uploader("Upload Spectral Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:", df.head())

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Ensure the target column 'vomitoxin_ppb' is removed from features
    if "vomitoxin_ppb" in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=["vomitoxin_ppb"])

    # Ensure only trained feature columns are selected
    missing_cols = [col for col in trained_feature_names if col not in df_numeric.columns]
    extra_cols = [col for col in df_numeric.columns if col not in trained_feature_names]

    if missing_cols:
        st.error(f"Error: The following required features are missing from your data: {missing_cols}")
    elif extra_cols:
        st.warning(f"Warning: The following extra columns were removed: {extra_cols}")
        df_numeric = df_numeric[trained_feature_names]  # Keep only trained feature columns

    # Apply the SAME SCALER and PCA as in training
    df_scaled = scaler_model.transform(df_numeric)
    df_pca = pca_model.transform(df_scaled)

    # Predict using XGBoost
    xgb_pred = xgb_model.predict(df_pca)

    # Display Result
    st.write("## Prediction:")
    st.write(f"**XGBoost Predicted DON Concentration:**")
    st.write(xgb_pred)

    # Scatter Plot - Predicted DON vs Sample Index
    st.write("### Scatter Plot: DON Prediction per Sample")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(xgb_pred)), xgb_pred, color="blue", alpha=0.7)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted DON (ppb)")
    ax.set_title("Scatter Plot of Predicted DON Concentration")
    st.pyplot(fig)

    # Line Plot - Trend of Predictions
    st.write("### Line Plot: Trend of Predicted DON Concentration")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(xgb_pred)), xgb_pred, marker="o", linestyle="-", color="green", alpha=0.7)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted DON (ppb)")
    ax.set_title("Line Plot Showing Trends in Predictions")
    st.pyplot(fig)

    # Allow user to download predictions
    pred_df = pd.DataFrame({"Predicted DON Concentration": xgb_pred})
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
