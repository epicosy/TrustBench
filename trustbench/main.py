from trustbench.core.sources.kaggle import Kaggle


if __name__ == '__main__':
    kaggle = Kaggle()
    ddir = kaggle.api.get_default_download_dir()
    print(ddir)
