import os
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

    def download(self, **kwargs):
        pass
