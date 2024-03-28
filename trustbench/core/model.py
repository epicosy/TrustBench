import keras
import numpy as np
import pandas as pd

from dataclasses import dataclass, field


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
