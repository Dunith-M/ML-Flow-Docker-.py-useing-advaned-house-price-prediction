import argparse

import joblib
import pandas as pd


class PredictionPipeline:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        transformed_data = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed_data)
        return float(prediction[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict house price")
    parser.add_argument("--area", type=int, required=True)
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=int, required=True)
    parser.add_argument("--stories", type=int, required=True)
    parser.add_argument("--mainroad", choices=["yes", "no"], required=True)
    parser.add_argument("--guestroom", choices=["yes", "no"], required=True)
    parser.add_argument("--basement", choices=["yes", "no"], required=True)
    parser.add_argument("--hotwaterheating", choices=["yes", "no"], required=True)
    parser.add_argument("--airconditioning", choices=["yes", "no"], required=True)
    parser.add_argument("--parking", type=int, required=True)
    parser.add_argument("--prefarea", choices=["yes", "no"], required=True)
    parser.add_argument(
        "--furnishingstatus",
        choices=["furnished", "semi-furnished", "unfurnished"],
        required=True,
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/model.pkl",
    )
    parser.add_argument(
        "--preprocessor-path",
        default="artifacts/preprocessor/preprocessor.pkl",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_data = {
        "area": args.area,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "stories": args.stories,
        "mainroad": args.mainroad,
        "guestroom": args.guestroom,
        "basement": args.basement,
        "hotwaterheating": args.hotwaterheating,
        "airconditioning": args.airconditioning,
        "parking": args.parking,
        "prefarea": args.prefarea,
        "furnishingstatus": args.furnishingstatus,
    }

    pipeline = PredictionPipeline(args.model_path, args.preprocessor_path)
    prediction = pipeline.predict(input_data)

    print("Prediction completed successfully.")
    print(f"Predicted house price: {prediction:.2f}")


if __name__ == "__main__":
    main()
