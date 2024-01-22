import json
from typing import Any
from pathlib import Path
from typing import Union


def read_config(config_file: Union[str, Path]):
    if isinstance(config_file, str):
        config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    if config_file.suffix != '.json':
        raise ValueError(f"Config file {config_file} must be a json file")

    with open(config_file, 'r') as f:
        return json.load(f)


def get_config(config: Any) -> dict:
    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        return read_config(config)
    elif isinstance(config, Path):
        return read_config(config)
    elif config is None:
        return {}

    raise ValueError(f"To be implemented config type: {type(config)}")


def get_kaggle_datasets(config_file: str):
    return read_config(config_file)
