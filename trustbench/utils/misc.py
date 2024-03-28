import json

from typing import Any, List, Dict
from pathlib import Path
from typing import Union
from tensorflow import keras

from trustbench.utils.paths import config_dir, data_dir, datasets_config_dir, models_dir, models_config_dir


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


def list_models() -> Dict[str, Path]:
    """
    List the models in the models directory
    :return: Dict of paths to the models
    """
    return {model.stem: model for model in models_dir.iterdir() if model.suffix == '.h5'}


def get_datasets_configs() -> dict:
    configs = {}

    for dc in datasets_config_dir.iterdir():
        if dc.is_file() and dc.suffix == '.json':
            configs.update(read_config(dc))

    return configs


def get_model_config(model: str) -> dict:
    model_config = models_config_dir / f"{model}.json"

    if not model_config.exists():
        raise ValueError(f"Model config {model_config} not found")

    return read_config(model_config)


def get_models_configs() -> dict:
    configs = {}

    for mc in models_config_dir.iterdir():
        if mc.is_file() and mc.suffix == '.json':
            configs.update(read_config(mc))

    return configs


def load_dataset(name: str, path: Path, config: dict):
    from trustbench.core.dataset import CSVDataset, NPYDataset
    preprocess_config = config.get('preprocess', {})

    if 'format' in config and config['format'] == 'npy':
        return NPYDataset(name=name, path=path, config=preprocess_config)
    else:
        return CSVDataset(name=name, path=path, config=preprocess_config)


def load_model(model: str) -> keras.Model:
    config = get_model_config(model)

    model_path = models_dir / config[model]['dataset']['name'] / f"{model}.h5"

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    return keras.models.load_model(str(model_path))
