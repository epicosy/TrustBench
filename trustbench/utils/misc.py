import json

from typing import Any, List, Dict
from pathlib import Path
from typing import Union

from trustbench.utils.paths import config_dir, data_dir


def get_configs():
    """
    Get all the configs from the config directory
    :return:
    """
    configs = {}

    for config in config_dir.iterdir():
        if config.suffix != '.json':
            continue

        configs.update(read_config(config))

    return configs


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


def find_config(search: str, exact: bool = False) -> Union[List[Path], Path]:
    results = []

    for file in config_dir.iterdir():
        if exact:
            if search == file.name:
                return file

        if search in file.name:
            results.append(file)

    return results


def find_sources():
    # find the sources in the core.sources package
    import importlib
    import pkgutil
    import trustbench.core.sources

    sources = {}

    for _, name, _ in pkgutil.iter_modules(trustbench.core.sources.__path__):
        module = importlib.import_module(f"trustbench.core.sources.{name}")

        if name == 'source':
            continue

        sources[name] = getattr(module, name.capitalize())()

    return sources


def list_datasets() -> Dict[str, Path]:
    """
    List the raw data in the data/raw directory
    :return: Dict of paths to the raw data of the respective dataset
    """

    return {dataset.name: dataset for dataset in data_dir.iterdir() if dataset.is_dir()}
