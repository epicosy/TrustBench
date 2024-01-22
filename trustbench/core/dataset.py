import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Split:
    name: str
    _features_file: str
    _labels_file: str
    headers: bool = True
    _features: pd.DataFrame = None
    _labels: np.ndarray = None
    _path: Path = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path):
        self._path = path / self.name

    @property
    def features(self):
        if self._features is None:
            self._features = pd.read_csv(str(self.path / self._features_file), delimiter=',', encoding='utf-8',
                                         header=None if not self.headers else 'infer')

        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.loadtxt(str(self.path / self._labels_file), dtype=int)
            #self._labels = pd.read_csv(str(self.path / self._labels_file), delimiter=',', encoding='utf-8', header=None,
            #                           names=['label'])

        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._features.to_csv(str(self.path / self._features_file), index=False, header=self.headers)
        np.savetxt(str(self.path / self._labels_file), self._labels, fmt='%d')


@dataclass
class Train(Split):
    name: str = 'train'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = False


@dataclass
class Val(Split):
    name: str = 'val'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = True


@dataclass
class Test(Split):
    name: str = 'test'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = True


class Dataset:
    def __init__(self, name: str, path: Path, config: dict = None):
        """

        :param name: Dataset name
        :param path: Path to the raw dataset
        """
        self.name = name
        self.root_path = path
        self.config = config
        self.path = [file for file in path.iterdir() if file.is_file() and file.suffix == '.csv'][0]
        self._df = None
        self.splits = {'train': Train(), 'val': Val(), 'test': Test()}

        for k, v in self.splits.items():
            v.path = self.root_path

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_csv(str(self.path), delimiter=',', encoding='utf-8', index_col=False)

        return self._df

    @df.setter
    def df(self, df):
        self._df = df

    def preprocess(self, scale: bool = False, encode: bool = False, binarize: bool = False):
        features = self.df.iloc[:, :-1]
        labels = self.df.iloc[:, -1]

        if encode:
            for col, encodings in self.config['encodings'].items():
                features[col] = features[col].map(encodings)

        if binarize:
            # TODO: find out why some categories are not encoded
            features = pd.get_dummies(features, drop_first=True)

        if scale:
            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            features = min_max_scaler.fit_transform(features)

        from sklearn.model_selection import train_test_split

        train_split = train_test_split(features, labels, test_size=0.2)

        train_set = self.splits['train']
        train_set.features, features_chunk, train_set.labels, labels_chunk = train_split
        train_set.save()

        val_test_split = train_test_split(features_chunk, labels_chunk, test_size=0.5)
        val_set = self.splits['val']
        test_set = self.splits['test']

        val_set.features, test_set.features, val_set.labels, test_set.labels = val_test_split
        val_set.save()
        test_set.save()
