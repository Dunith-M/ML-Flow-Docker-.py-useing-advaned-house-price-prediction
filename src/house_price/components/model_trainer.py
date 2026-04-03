import os
from pathlib import Path
from typing import Any

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

    def evaluate(self, y_test, y_pred) -> dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }

    def save_model(self, model: Any, output_path: str = "artifacts/models/model.pkl") -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        return output_path

    def get_models(self) -> dict[str, Any]:
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            ),
        }

    def get_model_params(self, model_name: str) -> dict[str, Any]:
        if model_name == "linear_regression":
            return {
                "model_name": "linear_regression",
                "fit_intercept": True,
            }

        if model_name == "random_forest":
            return {
                "model_name": "random_forest",
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "random_state": self.config.random_state,
            }

        return {"model_name": model_name}


def main() -> None:
    # -----------------------------
    # Production-style MLflow setup
    # -----------------------------
    tracking_dir = Path("mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.as_uri())

    experiment_name = "house_price_baseline"
    mlflow.set_experiment(experiment_name)

    # -----------------------------
    # Load configs and data
    # -----------------------------
    config = ConfigurationManager()
    transformation_config = config.get_data_transformation_config()
    trainer_config = config.get_model_trainer_config()

    transformer = DataTransformation(transformation_config)
    transformed_data = transformer.initiate_data_transformation()

    X_train = transformed_data["X_train"]
    y_train = transformed_data["y_train"]
    X_test = transformed_data["X_test"]
    y_test = transformed_data["y_test"]

    trainer = ModelTrainer(trainer_config)
    models = trainer.get_models()

    # -----------------------------
    # Best model tracking
    # -----------------------------
    best_model = None
    best_model_name = None
    best_metrics = None
    best_rmse = float("inf")

    model_output_dir = Path("artifacts/models")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Multi-run experiment tracking
    # -----------------------------
    for version, (model_name, model) in enumerate(models.items(), start=1):
        run_name = f"{model_name}_v{version}"
        model_params = trainer.get_model_params(model_name)

        with mlflow.start_run(run_name=run_name):
            # Structured tags
            mlflow.set_tags(
                {
                    "project": "house_price_prediction",
                    "stage": "training",
                    "run_type": "baseline_comparison",
                    "model_family": model_name,
                }
            )

            # Structured parameters
            mlflow.log_params(model_params)
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("run_name", run_name)
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            # Train
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Evaluate
            metrics = trainer.evaluate(y_test, predictions)

            # Log metrics
            mlflow.log_metrics(
                {
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "R2": metrics["R2"],
                }
            )

            # Save model artifact path per run
            model_file_path = model_output_dir / f"{model_name}.pkl"
            joblib.dump(model, model_file_path)

            mlflow.log_param("artifact_path", str(model_file_path))

            # Log model artifact to MLflow
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                )
            except Exception as exc:
                print(f"MLflow model logging skipped for {model_name}: {exc}")

            # Log local saved model file as an artifact too
            try:
                mlflow.log_artifact(str(model_file_path), artifact_path="saved_model_file")
            except Exception as exc:
                print(f"MLflow artifact file logging skipped for {model_name}: {exc}")

            # Best model selection
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = model
                best_model_name = model_name
                best_metrics = metrics

            print(f"\nRun Name: {run_name}")
            print(f"Model: {model_name}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"MAE: {metrics['MAE']:.4f}")
            print(f"R2: {metrics['R2']:.4f}")

    # -----------------------------
    # Save final best production model
    # -----------------------------
    if best_model is None or best_model_name is None or best_metrics is None:
        raise ValueError("No best model found.")

    final_model_path = trainer.save_model(
        best_model,
        output_path="artifacts/models/model.pkl",
    )

    # Separate run for best-model registration/logging
    with mlflow.start_run(run_name="best_model_summary"):
        mlflow.set_tags(
            {
                "project": "house_price_prediction",
                "stage": "best_model_selection",
                "selection_rule": "lowest_rmse",
            }
        )

        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("final_model_path", final_model_path)

        mlflow.log_metrics(
            {
                "best_RMSE": best_metrics["RMSE"],
                "best_MAE": best_metrics["MAE"],
                "best_R2": best_metrics["R2"],
            }
        )

        try:
            mlflow.log_artifact(final_model_path, artifact_path="best_model_file")
        except Exception as exc:
            print(f"Best model artifact logging skipped: {exc}")

        try:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="best_model",
            )
        except Exception as exc:
            print(f"Best model MLflow logging skipped: {exc}")

    print("\nBest model saved successfully.")
    print(f"Best Model: {best_model_name}")
    print(f"Path: {final_model_path}")
    print(f"Best RMSE: {best_metrics['RMSE']:.4f}")
    print(f"Best MAE: {best_metrics['MAE']:.4f}")
    print(f"Best R2: {best_metrics['R2']:.4f}")


if __name__ == "__main__":
    main()