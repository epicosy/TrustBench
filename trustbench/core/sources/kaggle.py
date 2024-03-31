import os
import shutil
import kagglehub

from pathlib import Path
from trustbench.core.sources.source import Source


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

    def get_dataset_ref(self, user: str, name: str):
        results = self.api.dataset_list(search=name, user=user)

        for result in results:
            if name in result.ref:
                return result

        raise ValueError(f"Dataset {name} not found for user {user}")

    def get_model_ref(self, owner: str, name: str):
        results = self.api.model_list(search=name, owner=owner)

        for result in results:
            if name in result.ref:
                return result

        raise ValueError(f"Model {name} not found for owner {owner}")

    def _download_dataset(self, name: str, owner: str, dataset_name: str):
        dataset = f"{owner}/{dataset_name}"
        path = self.data_dir / name
        path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset {dataset}")
        self.api.dataset_download_files(dataset=dataset, path=path, unzip=True)

    def _download_model(self, owner: str, name: str, framework: str, instance: str, version: str, file: str):
        model_path = self.models_dir / name
        model_path.mkdir(parents=True, exist_ok=True)
        handle = f"{owner}/{name}/{framework}/{instance}/{version}"

        model_file_path = model_path / file

        if model_file_path.exists():
            print(f"Model {handle} already downloaded")
            return

        try:
            cache_path = kagglehub.model_download(handle, force_download=True)
            print(f"Downloaded model to {cache_path}")
            output_file = Path(cache_path) / file
            # TODO: find a better way to do this
            print(f"Moving model to {model_path}")
            shutil.move(output_file, model_path)
        except Exception as e:
            print(f"Failed to download model {handle}: {e}")

    def download(self, name: str, **kwargs):
        if 'dataset_name' in kwargs:
            self._download_dataset(name, kwargs['owner'], kwargs['dataset_name'])

        if 'model_name' in kwargs:
            self._download_model(kwargs['owner'], kwargs['model_name'], kwargs['framework'], kwargs['instance'],
                                 kwargs['version'], file=kwargs['file'])
