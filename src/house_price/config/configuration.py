from pathlib import Path

from ..entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
)
from ..utils.common import read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_config_path: str = "configs/model.yaml",
        paths_config_path: str = "configs/paths.yaml",
    ) -> None:
        self.config = read_yaml(Path(config_path))
        self.model_config = read_yaml(Path(model_config_path))
        self.paths_config = read_yaml(Path(paths_config_path))

        self.schema_path = Path(self.paths_config["data_validation"]["schema_path"])
        self.schema = read_yaml(self.schema_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            raw_data_path=self.config["data_ingestion"]["raw_data_path"]
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_cfg = self.config["data_transformation"]

        return DataTransformationConfig(
            raw_data_path=self.config["data_ingestion"]["raw_data_path"],
            target_column=transformation_cfg["target_column"],
            test_size=transformation_cfg["test_size"],
            random_state=transformation_cfg["random_state"],
            preprocessor_path=transformation_cfg["preprocessor_path"],
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
            validated_data_path=self.paths_config["data_validation"][
                "validated_data_path"
            ],
            report_file_path=self.paths_config["data_validation"]["report_file_path"],
        )
