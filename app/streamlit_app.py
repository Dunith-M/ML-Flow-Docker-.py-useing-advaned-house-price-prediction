from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="House",
    layout="wide",
)

st.title("House Price Prediction System")

st.markdown(
    """
Welcome to the ML-powered house price prediction system.

Use the sidebar to navigate:
- Prediction
- Model Info
"""
)
