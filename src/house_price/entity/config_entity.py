from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str


@dataclass
class DataTransformationConfig:
    test_size: float
    random_state: int
    target_column: str


@dataclass
class ModelTrainerConfig:
    model_name: str
    n_estimators: int
    max_depth: int
    random_state: int
    
    
@dataclass
class DataValidationConfig:
    schema: dict
    raw_data_path: str
    validated_data_path: str
    report_file_path: str