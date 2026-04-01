from src.house_price.config.configuration import ConfigurationManager
from src.house_price.components.data_ingestion import DataIngestion
from src.house_price.components.data_validation import DataValidation
from src.house_price.utils.common import setup_logger


class Stage01DataPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        print("========== Stage 01: Data Ingestion Started ==========")

        # 🔹 Step 1: Data Ingestion
        ingestion_config = self.config_manager.get_data_ingestion_config()
        ingestion = DataIngestion(ingestion_config)

        raw_data_path = ingestion.run()

        print(f"Data Ingestion Completed. Output: {raw_data_path}")

        print("========== Data Validation Started ==========")

        # 🔹 Step 2: Data Validation
        validation_config = self.config_manager.get_data_validation_config()
        validation = DataValidation(validation_config)

        report = validation.run()

        print("Data Validation Completed.")
        print("Validation Report Summary:")
        print(report)

        print("========== Stage 01 Completed Successfully ==========")


if __name__ == "__main__":
    try:
        setup_logger()
        pipeline = Stage01DataPipeline()
        pipeline.run()
    except Exception as e:
        print(f"Error occurred in Stage 01 pipeline: {e}")
        raise e