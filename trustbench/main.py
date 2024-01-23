import argparse

from trustbench.utils.misc import find_sources, get_datasets_configs, list_datasets
from trustbench.core.dataset import Dataset


SOURCES = find_sources()
DATASETS_CONFIGS = get_datasets_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A Benchmark for Trustworthiness Evaluation of ML Models')

    subparsers = parser.add_subparsers(dest='subparser')
    collect_parser = subparsers.add_parser('collect')
    collect_parser.add_argument('-s', '--source', type=str, help='source', required=True,
                                choices=SOURCES.keys())

    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('-d', '--dataset', type=str, help='dataset', required=False,
                                   choices=DATASETS_CONFIGS.keys())

    args = parser.parse_args()

    if args.subparser == 'collect':
        source = SOURCES[args.source]

        for name, config in DATASETS_CONFIGS.items():
            collect_config = config.get('collect', {})

            if 'source' in collect_config and collect_config['source'] == args.source:
                source.download(name=name, **collect_config['kwargs'])

    elif args.subparser == 'preprocess':
        datasets_configs = {args.dataset: DATASETS_CONFIGS[args.dataset]} if args.dataset else DATASETS_CONFIGS
        datasets = list_datasets()

        for name, configs in datasets_configs.items():
            preprocess_config = configs.get('preprocess', {})

            if name not in datasets:
                continue

            dataset = Dataset(name, path=datasets[name], config=preprocess_config)
            dataset.preprocess()
