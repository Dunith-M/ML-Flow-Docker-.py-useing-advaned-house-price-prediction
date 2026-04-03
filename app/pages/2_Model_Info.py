from pathlib import Path

import joblib
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]

st.title("Model Information")

try:
    model = joblib.load(ROOT_DIR / "artifacts" / "models" / "model.pkl")

    st.subheader("Model Details")
    st.write(f"Model Type: {type(model).__name__}")

    if hasattr(model, "get_params"):
        st.subheader("Hyperparameters")
        st.json(model.get_params())

    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
