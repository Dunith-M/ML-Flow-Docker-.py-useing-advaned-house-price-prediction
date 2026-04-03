from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price.pipeline.prediction_pipeline import PredictionPipeline


st.title("Predict House Price")
st.subheader("Enter House Details")

area = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=1500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.number_input("Parking", min_value=0, max_value=5, value=1)
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"],
)

if st.button("Predict Price"):
    try:
        pipeline = PredictionPipeline(
            model_path=str(ROOT_DIR / "artifacts" / "models" / "model.pkl"),
            preprocessor_path=str(
                ROOT_DIR / "artifacts" / "preprocessor" / "preprocessor.pkl"
            ),
        )

        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }

        prediction = pipeline.predict(input_data)
        st.success(f"Estimated Price: {prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
