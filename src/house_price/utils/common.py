import yaml
from pathlib import Path


def read_yaml(path_to_yaml: Path) -> dict:
    try:
        with open(path_to_yaml, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise e