import os
from pathlib import Path

# get root_path from env variable
trustbench_path = os.environ.get('TRUSTBENCH_PATH', None)

if trustbench_path is None:
    raise ValueError("TRUSTBENCH_PATH environment variable is not set")

root_dir = Path(trustbench_path).expanduser().resolve()
root_dir.mkdir(parents=True, exist_ok=True)

config_dir = root_dir / 'config'
config_dir.mkdir(parents=True, exist_ok=True)

datasets_config_dir = config_dir / 'datasets'
datasets_config_dir.mkdir(parents=True, exist_ok=True)

data_dir = root_dir / 'data'
data_dir.mkdir(parents=True, exist_ok=True)

models_dir = root_dir / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

models_config_dir = config_dir / 'models'
models_config_dir.mkdir(parents=True, exist_ok=True)

predictions_dir = root_dir / 'predictions'
predictions_dir.mkdir(parents=True, exist_ok=True)

metadata_dir = root_dir / 'metadata'
metadata_dir.mkdir(parents=True, exist_ok=True)
