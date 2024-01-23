from pathlib import Path


root_dir = Path(__file__).parent.parent.parent
config_dir = root_dir / 'config'
config_dir.mkdir(parents=True, exist_ok=True)

datasets_config_dir = config_dir / 'datasets'
datasets_config_dir.mkdir(parents=True, exist_ok=True)

data_dir = root_dir / 'data'
data_dir.mkdir(parents=True, exist_ok=True)
