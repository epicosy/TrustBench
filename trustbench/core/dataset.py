from abc import abstractmethod

import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


@dataclass
class Split:
    name: str
    _features_file: str
    _labels_file: str
    headers: bool = True
    _features: pd.DataFrame = None
    _labels: np.ndarray = None
    _path: Path = None
    _format: str = 'csv'

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path):
        self._path = path / self.name

    @property
    def features(self):
        if self._features is None:
            if self._format == 'npy':
                self._features = np.load(str(self.path / self._features_file))
            else:
                self._features = pd.read_csv(str(self.path / self._features_file), delimiter=',', encoding='utf-8',
                                             header=None if not self.headers else 'infer')

        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def labels(self):
        if self._labels is None:
            if self._format == 'npy':
                self._labels = np.load(str(self.path / self._labels_file))
            else:
                self._labels = np.loadtxt(str(self.path / self._labels_file), dtype=int)

        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if self._format == 'csv':
            self._features.to_csv(str(self.path / self._features_file), index=False, header=self.headers)
            np.savetxt(str(self.path / self._labels_file), self._labels, fmt='%d')
        else:
            np.save(str(self.path / self._features_file), self._features)
            np.save(str(self.path / self._labels_file), self._labels)

    def resize(self, shape: tuple) -> np.ndarray:
        size = len(self.features)
        # resize it is for images so it should be 3 for the channels
        return self.features.to_numpy().reshape((size, *shape, 3))


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
    def __init__(self, name: str, path: Path, config: dict, data_format: str = 'csv'):
        """
        :param name: Dataset name
        :param path: Path to the raw dataset
        :param config: Dataset configuration
        :param format: File format of the dataset
        """
        self.name = name
        self.format = data_format
        self.root_path = path
        self.config = config
        self._data = None

        if self.format == 'csv':
            self.splits = {'train': Train(), 'val': Val(), 'test': Test()}
        else:
            self.splits = {'train': Train(_features_file='x.npy', _labels_file='y.npy', _format='npy'),
                           'val': Val(_features_file='x.npy', _labels_file='y.npy', _format='npy'),
                           'test': Test(_features_file='x.npy', _labels_file='y.npy', _format='npy')}

        for k, v in self.splits.items():
            v.path = self.root_path

    @property
    def data(self):
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data):
        pass

    @property
    def options(self):
        return self.config.get('options', {})

    @abstractmethod
    def preprocess(self):
        pass


class CSVDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(data_format='csv', **kwargs)
        self.path = [file for file in self.root_path.iterdir() if file.is_file() and file.suffix == '.csv'][0]

    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_csv(str(self.path), delimiter=',', encoding='utf-8', index_col=False)
            # drop any unnamed index column
            self._data = self._data.loc[:, ~self._data.columns.str.contains('^Unnamed')]

        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def _transform(self):
        from pandas.api.types import is_string_dtype
        encodings = self.options.get('encode', {})

        if not isinstance(encodings, dict):
            raise ValueError("Encodings must be a dictionary of column names to encodings. e.g. "
                             "{'col1': {'a': 0, 'b': 1}}")

        binarize = self.options.get('binarize', False)
        scale = self.options.get('scale', False)

        # TODO: this it is slow for large datasets; consider using a more efficient method
        for col in tqdm(self.data.columns):
            if col in encodings:
                print(f"Encoding column {col}")
                self.data[col] = self.data[col].map(encodings[col])
            if is_string_dtype(self.data[col]):
                print(f"Binarizing column {col}")
                if binarize:
                    # the other categorical columns should be one-hot encoded
                    self.data = pd.concat([self.data, pd.get_dummies(self.data[col], prefix=col)], axis=1)
                    self.data.drop(col, axis=1, inplace=True)
            else:
                if scale:
                    print(f"Scaling column {col}")
                    from sklearn.preprocessing import MinMaxScaler

                    min_max_scaler = MinMaxScaler()
                    self.data[col] = min_max_scaler.fit_transform(self.data[col].values.reshape(-1, 1))

    def _clip(self, features: pd.DataFrame, max_clip: float) -> pd.DataFrame:
        # TODO: why only use clip_max?
        #  https://github.com/self-checker/SelfChecker/blob/master/main_kde.py#L111

        if max_clip:
            # check the range for the clip
            if max_clip < 0 or max_clip > 1:
                raise ValueError("Clip range must be between 0 and 1.")

            # get all the columns that are not categorical
            non_categorical = features.select_dtypes(exclude=['object']).columns

            features[non_categorical] = features[non_categorical].astype("float32")
            features[non_categorical] = features[non_categorical].applymap(lambda x: (x / 255.0) - (1.0 - max_clip))

            return features

    def preprocess(self):
        from sklearn.model_selection import train_test_split

        self._transform()
        labels_col = self.config.get('labels_col', None)

        labels = self.data[labels_col]
        features = self.data.drop(labels_col, axis=1)

        clip = self.options.get('clip', {})

        if clip:
            clip_max = clip.get('max', None)
            if clip_max:
                features = self._clip(features, clip_max)
            else:
                raise ValueError("Both min and max must be provided for clipping.")

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

    def get_resize_shape(self) -> tuple:
        resize_config = self.config.get('options', {}).get('resize', {})

        if resize_config:
            return resize_config['width'], resize_config['height']

        else:
            return ()


class NPYDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(data_format='npy', **kwargs)
        self.path = [file for file in self.root_path.iterdir() if file.is_file() and file.suffix == '.npz'][0]

    @property
    def data(self):
        if self._data is None:
            self._data = np.load(str(self.path))

        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def _clip(self, x: np.ndarray) -> np.ndarray:
        clip = self.options.get('clip', {})

        if clip:
            clip_max = clip.get('max', None)
            if clip_max:
                # check the range for the clip
                if clip_max < 0 or clip_max > 1:
                    raise ValueError("Clip range must be between 0 and 1.")

                return (x.astype("float32") / 255.0) - (1.0 - clip_max)
            else:
                raise ValueError("Both min and max must be provided for clipping.")

        return x

    def _split(self, x: np.ndarray, y: np.ndarray) -> tuple:
        split = self.options.get('split', {})

        if split:
            num_train = split.get('train', None)
            if num_train:
                # split original training dataset into training and validation dataset
                x_train = x[:num_train]
                x_valid = x[num_train:]
                y_train = y[:num_train]
                y_valid = y[num_train:]
                return x_train, x_valid, y_train, y_valid
            else:
                raise ValueError("Number of training samples must be provided for splitting.")

        return x, y, None, None

    def preprocess(self):
        x_train_total = self._clip(self.data['x_train'])
        y_train_total = self.data['y_train'].reshape([self.data['y_train'].shape[0]])

        (self.splits['train'].features, self.splits['val'].features,
         self.splits['train'].labels, self.splits['val'].labels) = self._split(x_train_total, y_train_total)

        self.splits['test'].features = self._clip(self.data['x_test'])
        self.splits['test'].labels = self.data['y_test'].reshape([self.data['y_test'].shape[0]])

        # split original training dataset into training and validation dataset
        print("x_train len:{}".format(len(self.splits['train'].features)))
        print("x_valid len:{}".format(len(self.splits['val'].features)))
        print("x_test len:{}".format(len(self.splits['test'].features)))
        print("y_train len:{}".format(len(self.splits['train'].labels)))
        print("y_valid len:{}".format(len(self.splits['val'].labels)))
        print("y_test len:{}".format(len(self.splits['test'].labels)))

        self.splits['test'].save()
        self.splits['train'].save()
        self.splits['val'].save()
