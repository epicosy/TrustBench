from trustbench.core.sources.kaggle import Kaggle
from trustbench.utils.misc import get_kaggle_datasets
from trustbench.utils.paths import config_dir

if __name__ == '__main__':
    kaggle = Kaggle()

    for name, dataset in get_kaggle_datasets(config_file=config_dir / 'kaggle_datasets.json').items():
        kaggle.download(**dataset)
