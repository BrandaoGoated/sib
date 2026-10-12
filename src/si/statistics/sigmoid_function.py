import numpy as np


def sigmoid_function(X):
    """
    Calculate the sigmoid function element-wise.

    Parameters
    ----------
    X : array-like
        Input values.

    Returns
    -------
    numpy.ndarray
        Probabilities between 0 and 1.
    """
    X = np.asarray(X, dtype=float)

    result = np.empty_like(X)
    positive = X >= 0

    result[positive] = 1.0 / (1.0 + np.exp(-X[positive]))

    exp_X = np.exp(X[~positive])
    result[~positive] = exp_X / (1.0 + exp_X)

    return result