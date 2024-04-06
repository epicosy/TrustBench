import argparse

from trustbench.utils.misc import (find_sources, get_datasets_configs, get_dataset, get_models_configs)
from trustbench.core.model import get_metadata

SOURCES = find_sources()
DATASETS_CONFIGS = get_datasets_configs()
MODELS_CONFIGS = get_models_configs()


def main():
    parser = argparse.ArgumentParser(description='A Benchmark for Trustworthiness Evaluation of ML Models')

    subparsers = parser.add_subparsers(dest='subparser')
    collect_parser = subparsers.add_parser('collect')
    collect_parser.add_argument('-s', '--source', type=str, help='source', required=True,
                                choices=SOURCES.keys())
    collect_parser.add_argument('-d', '--datasets', action='store_true', help='gets datasets', required=False)
    collect_parser.add_argument('-m', '--models', action='store_true', help='gets models', required=False)

    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('-d', '--dataset', type=str, help='dataset', required=False,
                                   choices=DATASETS_CONFIGS.keys())
    preprocess_parser.add_argument('-rs', '--random_state', type=int, required=False, default=42,
                                   help='Random state for reproducibility')
    # TODO: add argument for preprocessing all datasets for a given source

    detail_parser = subparsers.add_parser('detail')
    detail_parser.add_argument('-d', '--dataset', type=str, help='dataset', required=True,
                               choices=DATASETS_CONFIGS.keys())
    detail_parser.add_argument('-m', '--model', type=str, help='model', required=False,
                               choices=MODELS_CONFIGS.keys())

    args = parser.parse_args()

    if args.subparser == 'collect':
        source = SOURCES[args.source]
        to_collect = []

        if args.datasets:
            to_collect.extend(DATASETS_CONFIGS.items())

        if args.models:
            to_collect.extend(MODELS_CONFIGS.items())

        for name, config in to_collect:
            collect_config = config.get('collect', {})

            if 'source' in collect_config and collect_config['source'] == args.source:
                source.download(name=name, **collect_config['kwargs'])

    elif args.subparser == 'preprocess':
        datasets_configs = {args.dataset: DATASETS_CONFIGS[args.dataset]} if args.dataset else DATASETS_CONFIGS

        for name, configs in datasets_configs.items():
            dataset = get_dataset(name=name)
            dataset.preprocess(args.random_state)

    elif args.subparser == 'detail':
        models_configs = {args.model: MODELS_CONFIGS[args.model]} if args.model else MODELS_CONFIGS

        for name, configs in models_configs.items():
            if args.dataset != configs['dataset']['name']:
                # print(f'Model: {name} is not for dataset: {args.dataset}')
                continue

            metadata = get_metadata(model_name=name, dataset_name=args.dataset)

            print(name, metadata)


if __name__ == '__main__':
    main()
