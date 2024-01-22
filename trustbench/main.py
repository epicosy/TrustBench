import argparse

from trustbench.utils.misc import find_sources, list_datasets, get_configs
from trustbench.core.dataset import Dataset


SOURCES = find_sources()
DATASETS = list_datasets()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A Benchmark for Trustworthiness Evaluation of ML Models')

    subparsers = parser.add_subparsers(dest='subparser')
    collect_parser = subparsers.add_parser('collect')
    collect_parser.add_argument('-s', '--source', type=str, help='source', required=True,
                                choices=SOURCES.keys())

    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('-d', '--dataset', type=str, help='dataset', required=False,
                                   choices=DATASETS.keys())
    preprocess_parser.add_argument('-e', '--encode', action='store_true', default=False,
                                   help='Encode the columns with the specified values in the config.')
    preprocess_parser.add_argument('-b', '--binarize', action='store_true', default=False,
                                   help='Convert categorical variable into dummy/indicator variables.')

    args = parser.parse_args()

    if args.subparser == 'collect':
        source = SOURCES[args.source]

        for name, dataset in source.list_datasets():
            source.download(name=name, **dataset)

    elif args.subparser == 'preprocess':
        datasets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
        configs = get_configs()

        for name, path in datasets.items():
            config = configs.get(name, None)
            dataset = Dataset(name, path, config)
            dataset.preprocess(encode=args.encode, binarize=args.binarize)
