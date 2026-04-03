from src.house_price.components.data_ingestion import DataIngestion
from src.house_price.components.data_validation import DataValidation
from src.house_price.components.data_transformation import DataTransformation
from src.house_price.components.model_trainer import ModelTrainer
from src.house_price.config.configuration import ConfigurationManager


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        print(">>> Stage 1: Data Ingestion")
        ingestion_config = self.config.get_data_ingestion_config()
        ingestion = DataIngestion(ingestion_config)
        data_path = ingestion.initiate_data_ingestion()

        print(">>> Stage 2: Data Validation")
        validation_config = self.config.get_data_validation_config()
        validation = DataValidation(validation_config)
        validation.validate(data_path)

        print(">>> Stage 3: Data Transformation")
        transformation_config = self.config.get_data_transformation_config()
        transformation = DataTransformation(transformation_config)
        X_train, X_test, y_train, y_test, preprocessor_path = (
            transformation.initiate_data_transformation(data_path)
        )

        print(">>> Stage 4: Model Training")
        trainer_config = self.config.get_model_trainer_config()
        trainer = ModelTrainer(trainer_config)
        model_path = trainer.train(X_train, y_train, X_test, y_test)

        print(">>> Stage 5: Evaluation Completed")

        return {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
        }