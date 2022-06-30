from pathlib import Path
import yaml
from src.config.config import Config
import json


def load_yaml_config_from_path(filepath: Path) -> Config:
    config_yaml = yaml.safe_load(filepath.read_bytes())
    config = Config(**config_yaml)
    return config


def load_json_config_from_path(filepath: Path) -> Config:
    with open(filepath, 'r') as file:
        config_json = json.load(file)
    config = Config(**config_json)
    return config
