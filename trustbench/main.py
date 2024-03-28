import argparse

import pandas as pd

from trustbench.utils.misc import (find_sources, get_datasets_configs, list_datasets, load_dataset, load_model,
                                   get_models_configs)

from trustbench.utils.paths import predictions_dir
from trustbench.core.model import predict_unseen

SOURCES = find_sources()
DATASETS_CONFIGS = get_datasets_configs()
MODELS_CONFIGS = get_models_configs()


def main():
    parser = argparse.ArgumentParser(description='A Benchmark for Trustworthiness Evaluation of ML Models')

    subparsers = parser.add_subparsers(dest='subparser')
    collect_parser = subparsers.add_parser('collect')
    collect_parser.add_argument('-s', '--source', type=str, help='source', required=True,
                                choices=SOURCES.keys())

    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('-d', '--dataset', type=str, help='dataset', required=False,
                                   choices=DATASETS_CONFIGS.keys())

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('-d', '--dataset', type=str, help='dataset', required=True,
                                choices=DATASETS_CONFIGS.keys())
    predict_parser.add_argument('-m', '--model', type=str, help='model', required=False,
                                choices=MODELS_CONFIGS.keys())

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
            if name not in datasets:
                continue

            dataset = load_dataset(name=name, path=datasets[name], config=configs)
            dataset.preprocess()

    elif args.subparser == 'predict':
        datasets = list_datasets()
        dataset = load_dataset(name=args.dataset, path=datasets[args.dataset], config=DATASETS_CONFIGS[args.dataset])
        models_configs = {args.model: MODELS_CONFIGS[args.model]} if args.model else MODELS_CONFIGS

        for name, configs in models_configs.items():
            if args.dataset != configs['dataset']['name']:
                print(f'Model: {name} is not for dataset: {args.dataset}')
                continue

            print(f'Predicting for model: {name}')
            model = load_model(model=name)
            predictions = predict_unseen(model, features=dataset.splits['test'].features,
                                         labels=dataset.splits['test'].labels)

            print(f'Acc:', predictions.correct/len(predictions.labels))
            predictions_path = predictions_dir / args.dataset
            predictions_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'y': predictions.labels}).to_csv(predictions_path / f"{name}.csv", index=False)


if __name__ == '__main__':
    main()
