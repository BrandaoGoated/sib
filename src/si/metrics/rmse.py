import numpy as np


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error.
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

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )