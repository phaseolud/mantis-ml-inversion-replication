from pathlib import Path
import yaml
from src.config.config import Config


def load_config_from_path(filepath: Path) -> Config:
    config_yaml = yaml.safe_load(filepath.read_bytes())
    config = Config(**config_yaml)
    return config
