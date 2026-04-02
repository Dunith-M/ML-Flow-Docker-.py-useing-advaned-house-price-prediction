from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataIngestionConfig:
    raw_data_path: str


@dataclass
class ModelTrainerConfig:
    model_name: str
    n_estimators: int
    max_depth: int
    random_state: int
    
    
@dataclass
class DataValidationConfig:
    schema: Dict[str, Any]
    raw_data_path: str
    validated_data_path: str
    report_file_path: str
    

@dataclass
class DataTransformationConfig:
    raw_data_path: str
    target_column: str
    test_size: float
    random_state: int
    preprocessor_path: str
