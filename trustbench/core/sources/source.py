from abc import abstractmethod

from trustbench.utils.misc import get_config
from trustbench.utils.paths import data_dir


class Source:
    def __init__(self, name: str, config=None, **kwargs):
        self.name = name
        # by default, pass the path to the config file with the name of the source
        self.config = get_config(config)
        self._api = None
        self.data_dir = data_dir
        self.init(**kwargs)

    @property
    @abstractmethod
    def api(self):
        pass

    @abstractmethod
    def init(self, **kwargs):
        pass

    @abstractmethod
    def download(self, name: str, **kwargs):
        pass

    @abstractmethod
    def list_datasets(self, **kwargs) -> tuple:
        pass
