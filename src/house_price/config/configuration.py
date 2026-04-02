from pathlib import Path

from ..utils.common import read_yaml
from ..entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    DataValidationConfig,
    DataTransformationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_path="configs/config.yaml",
        model_config_path="configs/model.yaml",
        paths_config_path="configs/paths.yaml",
    ):
        self.config = read_yaml(Path(config_path))
        self.model_config = read_yaml(Path(model_config_path))
        self.paths_config = read_yaml(Path(paths_config_path))

        # Better: get schema path from config (not hardcoded)
        self.schema_path = Path(self.paths_config["data_validation"]["schema_path"])
        self.schema = read_yaml(self.schema_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            raw_data_path=self.config["data_ingestion"]["raw_data_path"]
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            test_size=self.config["data_transformation"]["test_size"],
            random_state=self.config["data_transformation"]["random_state"],
            target_column=self.config["data_transformation"]["target_column"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            model_name=self.model_config["model_trainer"]["model_name"],
            n_estimators=self.model_config["model_trainer"]["n_estimators"],
            max_depth=self.model_config["model_trainer"]["max_depth"],
            random_state=self.model_config["model_trainer"]["random_state"],
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        return DataValidationConfig(
            schema=self.schema,
            raw_data_path=self.config["data_ingestion"]["raw_data_path"],
            validated_data_path=self.paths_config["data_validation"]["validated_data_path"],
            report_file_path=self.paths_config["data_validation"]["report_file_path"],
        )
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            raw_data_path=self.config["data_ingestion"]["raw_data_path"],
            target_column=self.config["data_transformation"]["target_column"],
            test_size=self.config["data_transformation"]["test_size"],
            random_state=self.config["data_transformation"]["random_state"],
    )