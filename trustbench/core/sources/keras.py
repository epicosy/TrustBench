import numpy as np
import pandas as pd

from trustbench.core.sources.source import Source


class Keras(Source):
    def __init__(self):
        super().__init__(name='keras')

    @property
    def api(self):
        if self._api is None:
            from keras import datasets
            self._api = datasets

        return self._api

    def init(self, **kwargs):
        pass

    def download(self, name: str, **kwargs):
        import importlib

        path = self.data_dir / name
        module_path = f"{self.api.__name__}.{kwargs['module_name']}"
        path.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(module_path)
        (x_train, y_train), (x_test, y_test) = module.load_data()
        # concatenate the x_train and x_test
        features = np.concatenate((x_train, x_test))
        labels = np.concatenate((y_train, y_test))

        # flatten each image
        features = np.array([x.flatten() for x in features])
        # generate column names for the features
        columns = [f"pixel_{i}" for i in range(features.shape[1])]
        # create a dataframe with the features
        df = pd.DataFrame(features, columns=columns)
        # add the labels to the dataframe
        df['label'] = labels

        # randomly select 80% of the data for training
        df = df.sample(frac=0.05, random_state=200)

        # save the data
        df.to_csv(str(path / f'{name}.csv'), index=False)
