import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config.configuration import ConfigurationManager
from .data_transformation import DataTransformation


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_score = float("inf")

    def evaluate(self, y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }

    def save_model(self, model, output_path: str = "artifacts/models/model.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        return output_path


def main() -> None:
    # Setup tracking
    tracking_dir = Path("mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("house_price_experiment")

    # Load configs
    config = ConfigurationManager()
    transformation_config = config.get_data_transformation_config()
    trainer_config = config.get_model_trainer_config()

    # Data
    transformer = DataTransformation(transformation_config)
    transformed_data = transformer.initiate_data_transformation()

    X_train = transformed_data["X_train"]
    y_train = transformed_data["y_train"]
    X_test = transformed_data["X_test"]
    y_test = transformed_data["y_test"]

    trainer = ModelTrainer(trainer_config)

    # MULTI-MODEL SETUP
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=trainer_config.n_estimators,
            max_depth=trainer_config.max_depth,
            random_state=trainer_config.random_state,
        ),
    }

    best_model = None
    best_rmse = float("inf")

    # LOOP OVER MODELS
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            # Log params
            mlflow.log_param("model_name", model_name)

            if model_name == "random_forest":
                mlflow.log_param("n_estimators", trainer_config.n_estimators)
                mlflow.log_param("max_depth", trainer_config.max_depth)
                mlflow.log_param("random_state", trainer_config.random_state)

            # Train
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Evaluate
            metrics = trainer.evaluate(y_test, predictions)

            # Log metrics
            mlflow.log_metric("RMSE", metrics["RMSE"])
            mlflow.log_metric("MAE", metrics["MAE"])
            mlflow.log_metric("R2", metrics["R2"])

            # Log model
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                )
            except Exception as exc:
                print(f"MLflow model logging skipped: {exc}")

            # SELECT BEST MODEL
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = model

            print(f"\nModel: {model_name}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"MAE: {metrics['MAE']:.4f}")
            print(f"R2: {metrics['R2']:.4f}")

    # SAVE BEST MODEL ONLY
    if best_model is None:
        raise ValueError("No best model found.")

    final_model_path = trainer.save_model(best_model)

    print("\nBest model saved.")
    print(f"Path: {final_model_path}")
    print(f"Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()