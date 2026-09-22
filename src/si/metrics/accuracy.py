import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate the proportion of correctly predicted labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "y_true and y_pred cannot be empty."
        )

    return np.sum(y_true == y_pred) / y_true.size