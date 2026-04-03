from src.house_price.pipeline.training_pipeline import TrainingPipeline
from src.house_price.pipeline.prediction_pipeline import PredictionPipeline


def main() -> None:
    try:
        print("===== TRAINING PIPELINE STARTED =====")
        training_pipeline = TrainingPipeline()
        artifacts = training_pipeline.run()

        print("===== TRAINING COMPLETED =====")

        print("===== PREDICTION PIPELINE STARTED =====")

        prediction_pipeline = PredictionPipeline(
            model_path=artifacts["model_path"],
            preprocessor_path=artifacts["preprocessor_path"],
        )

        #  Replace with real schema-aligned input
        sample_input = {
            "area": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
        }

        prediction = prediction_pipeline.predict(sample_input)

        print(f"Prediction: {prediction}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()