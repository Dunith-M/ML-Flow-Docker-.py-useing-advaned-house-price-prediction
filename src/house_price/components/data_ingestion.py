import os
from pathlib import Path
import pandas as pd

from ..config.configuration import ConfigurationManager
from ..entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def read_data(self) -> pd.DataFrame:
        """
        Step 1: Read raw CSV from data/raw/
        """

        file_path = Path(self.config.raw_data_path)

        #  Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found at: {file_path}")

        #  Load CSV
        df = pd.read_csv(file_path)

        return df

    def save_to_interim(self, df: pd.DataFrame) -> str:
        """
        Step 2: Save a copy into data/interim/
        """

        # Define output path
        interim_path = Path("data/interim/raw_loaded.csv")

        # Create directory if not exists
        interim_path.parent.mkdir(parents=True, exist_ok=True)

        # Save file
        df.to_csv(interim_path, index=False)

        return str(interim_path)

    def run(self) -> str:
        """
        Full ingestion pipeline:
        - read raw data
        - save interim copy
        """

        df = self.read_data()
        saved_path = self.save_to_interim(df)

        return saved_path


def main() -> None:
    config = ConfigurationManager()
    ingestion_config = config.get_data_ingestion_config()
    ingestion = DataIngestion(ingestion_config)
    saved_path = ingestion.run()
    print(f"Data ingestion completed. File saved to: {saved_path}")


if __name__ == "__main__":
    main()
