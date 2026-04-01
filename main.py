from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price.config.configuration import ConfigurationManager

config = ConfigurationManager()

data_config = config.get_data_ingestion_config()
model_config = config.get_model_trainer_config()

print(data_config)
print(model_config)
