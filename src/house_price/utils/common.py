import yaml
import logging
from pathlib import Path
from datetime import datetime


# ----------------------------
# 1. Logging Setup
# ----------------------------
def setup_logger(log_dir: str = "artifacts/logs") -> None:
    """
    Initialize logging configuration
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ----------------------------
# 2. YAML Reader (Improved)
# ----------------------------
def read_yaml(path_to_yaml: Path) -> dict:
    try:
        if not path_to_yaml.exists():
            raise FileNotFoundError(f"YAML file not found: {path_to_yaml}")

        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        logging.info(f"YAML file loaded successfully: {path_to_yaml}")
        return content

    except Exception as e:
        raise CustomException(e)


# ----------------------------
# 3. Custom Exception Wrapper
# ----------------------------
class CustomException(Exception):
    def __init__(self, error_message: Exception):
        super().__init__(str(error_message))
        self.error_message = str(error_message)

    def __str__(self):
        return f"Pipeline Exception: {self.error_message}"