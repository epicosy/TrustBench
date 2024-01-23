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
    def __init__(self, name: str, path: Path, config: dict):
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
    def options(self):
        return self.config.get('options', {})

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_csv(str(self.path), delimiter=',', encoding='utf-8', index_col=False)
            # drop any unnamed index column
            self._df = self._df.loc[:, ~self._df.columns.str.contains('^Unnamed')]

        return self._df

    @df.setter
    def df(self, df):
        self._df = df

    def _transform(self):
        from pandas.api.types import is_string_dtype
        encodings = self.options.get('encode', {})

        if not isinstance(encodings, dict):
            raise ValueError("Encodings must be a dictionary of column names to encodings. e.g. "
                             "{'col1': {'a': 0, 'b': 1}}")

        binarize = self.options.get('binarize', False)
        scale = self.options.get('scale', False)

        for col in self.df.columns:
            if col in encodings:
                print(f"Encoding column {col}")
                self.df[col] = self.df[col].map(encodings[col])
            if is_string_dtype(self.df[col]):
                print(f"Binarizing column {col}")
                if binarize:
                    # the other categorical columns should be one-hot encoded
                    self.df = pd.concat([self.df, pd.get_dummies(self.df[col], prefix=col)], axis=1)
                    self.df.drop(col, axis=1, inplace=True)
            else:
                if scale:
                    print(f"Scaling column {col}")
                    from sklearn.preprocessing import MinMaxScaler

                    min_max_scaler = MinMaxScaler()
                    self.df[col] = min_max_scaler.fit_transform(self.df[col].values.reshape(-1, 1))

    def preprocess(self):
        from sklearn.model_selection import train_test_split

        self._transform()
        labels_col = self.config.get('labels_col', None)

        labels = self.df[labels_col]
        features = self.df.drop(labels_col, axis=1)

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
