import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config.configuration import ConfigurationManager
from .data_transformation import DataTransformation


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = self._get_model()
        self.best_model = None
        self.best_score = float("inf")

    def _get_model(self):
        if self.config.model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            )

        return LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }

    def select_best_model(self, metrics):
        if metrics["RMSE"] < self.best_score:
            self.best_score = metrics["RMSE"]
            self.best_model = self.model

    def save_model(self, output_path: str = "artifacts/models/model.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.best_model is None:
            raise ValueError("No model selected to save.")

        joblib.dump(self.best_model, output_path)
        return output_path


def main() -> None:
    config = ConfigurationManager()
    transformation_config = config.get_data_transformation_config()
    trainer_config = config.get_model_trainer_config()

    transformer = DataTransformation(transformation_config)
    transformed_data = transformer.initiate_data_transformation()

    trainer = ModelTrainer(trainer_config)
    trainer.train(transformed_data["X_train"], transformed_data["y_train"])
    predictions = trainer.predict(transformed_data["X_test"])

    metrics = trainer.evaluate(transformed_data["y_test"], predictions)
    trainer.select_best_model(metrics)
    model_path = trainer.save_model()

    print("Model training completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
