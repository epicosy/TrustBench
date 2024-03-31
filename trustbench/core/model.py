import json
import keras
import numpy as np
import pandas as pd


from dataclasses import dataclass, field
from trustbench.utils.misc import load_model, get_dataset
from trustbench.utils.paths import predictions_dir, metadata_dir


@dataclass
class Predictions:
    labels: list = field(default_factory=lambda: [])
    correct: int = 0
    incorrect: int = 0


# needs to be updated to accommodate different types
def predict_unseen(model: keras.Model, features: pd.DataFrame, labels: pd.DataFrame) -> Predictions:
    unseen_ops = model.predict(features)
    predictions = Predictions()

    if len(unseen_ops[0]) == 1:
        cnt_0 = 0
        # TODO: why this count is set to one?
        cnt_1 = 1

        for i in range(0, len(unseen_ops)):
            if unseen_ops[i][0] > 0.5:
                cnt_1 += + 1
                predictions.labels.append(1)

                if labels.loc[i].item() == 1:
                    predictions.correct += 1
                else:
                    predictions.incorrect += 1
            else:
                cnt_0 += 1
                predictions.labels.append(0)
                if labels.loc[i].item() == 0:
                    predictions.correct += 1
                else:
                    predictions.incorrect += 1

        print("UNSEEN: Label 0:", cnt_0, "Label 1:", cnt_1)
        print("UNSEEN: ACT CORR:", predictions.correct, ", ACT INCORR:", predictions.incorrect)
    else:
        # perform multi-class classification
        for i in range(0, len(unseen_ops)):
            prediction = np.argmax(unseen_ops[i])
            predictions.labels.append(prediction)

            if labels[i] == prediction:
                predictions.correct += 1
            else:
                predictions.incorrect += 1

        # get labels count
        unique, counts = np.unique(predictions.labels, return_counts=True)
        counts_str = "UNSEEN: "

        for i in range(0, len(unique)):
            counts_str += f"Label {unique[i]}: {counts[i]}, "

        print(counts_str)
        print("UNSEEN: ACT CORR:", predictions.correct, ", ACT INCORR:", predictions.incorrect)

    return predictions


def get_metadata(model_name: str, dataset_name: str) -> dict:
    dataset = get_dataset(name=dataset_name)
    predictions_path = predictions_dir / dataset_name / f"{model_name}.csv"
    metadata_path = metadata_dir / dataset_name / f"{model_name}.json"

    if not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        model = load_model(model=model_name)

        if not predictions_path.exists():
            predictions_path.parent.mkdir(parents=True, exist_ok=True)

            print(f'Predicting for model: {model_name}')
            predictions = predict_unseen(model, features=dataset.splits['test'].features,
                                         labels=dataset.splits['test'].labels)

            df = pd.DataFrame({'y': predictions.labels})
            df.to_csv(predictions_path, index=False)

        else:
            df = pd.read_csv(predictions_path)
            df = df.merge(dataset.splits['test'].labels, left_index=True, right_index=True, how='inner',
                          suffixes=('_pred', '_true'))
            predictions = Predictions()

            for _, row in df.iterrows():
                predictions.labels.append(row['y_pred'])
                if row['y_pred'] == row['y_true']:
                    predictions.correct += 1
                else:
                    predictions.incorrect += 1

        metadata = {
            'predictions': {
                'correct': predictions.correct,
                'incorrect': predictions.incorrect,
                'total': len(predictions.labels),
            },
            'model': {
                '#layers': len(model.layers),
                "#params": model.count_params(),
            },
            "accuracy": round((predictions.correct / len(predictions.labels)) * 100, 2),
        }

        with metadata_path.open(mode='w') as f:
            json.dump(metadata, f, indent=4)
    else:
        with metadata_path.open(mode='r') as f:
            metadata = json.load(f)

    return metadata
