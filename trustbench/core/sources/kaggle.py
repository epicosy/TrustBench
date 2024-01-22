import os

from trustbench.core.sources.source import Source
from trustbench.utils.misc import find_config, read_config


class Kaggle(Source):
    def __init__(self):
        super().__init__(name='kaggle')

    @property
    def api(self):
        if self._api is None:
            from kaggle import api
            self._api = api

        return self._api

    def init(self, **kwargs):
        os.environ['KAGGLE_PATH'] = str(self.data_dir)

    def list_datasets(self, **kwargs) -> tuple:
        config_file = find_config('kaggle_datasets.json', exact=True)

        if isinstance(config_file, list):
            raise FileNotFoundError(f"Config file kaggle_datasets.json not found")

        return read_config(config_file).items()

    def get_dataset_ref(self, user: str, name: str):
        results = self.api.dataset_list(search=name, user=user)

        for result in results:
            if name in result.ref:
                return result

        raise ValueError(f"Dataset {name} not found for user {user}")

    def download(self, name: str, **kwargs):
        dataset = f"{kwargs['owner']}/{kwargs['dataset_name']}"
        path = self.data_dir / name
        path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset {dataset}")
        self.api.dataset_download_files(dataset=dataset, path=path, unzip=True)
