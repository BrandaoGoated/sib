import numpy as np

from ..base.model import Model
from ..metrics.rmse import rmse


def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two samples.
    """
    return np.sqrt(
        np.sum(
            (x1 - x2) ** 2
        )
    )


class KNNRegressor(Model):
    """
    K-Nearest Neighbors regressor.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        """
        Initialize the regressor.

        Parameters
        ----------
        k : int, default=5
            Number of nearest neighbours used for regression.

        distance : callable, default=euclidean_distance
            Function used to calculate distances between samples.
        """
        super().__init__()

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset):
        """
        Store the training Dataset.
        """
        if not dataset.has_label():
            raise ValueError(
                "KNNRegressor requires a Dataset with labels."
            )

        if self.k > dataset.X.shape[0]:
            raise ValueError(
                "k cannot be greater than the number of training samples."
            )

        self.dataset = dataset

        return self

    def _predict(self, dataset):
        """
        Predict the target value for each sample in a Dataset.
        """
        predictions = []

        for sample in dataset.X:
            distances = np.array([
                self.distance(sample, train_sample)
                for train_sample in self.dataset.X
            ])

            nearest_indices = np.argsort(distances)[:self.k]

            nearest_values = self.dataset.y[nearest_indices]

            predicted_value = np.mean(
                nearest_values.astype(float)
            )

            predictions.append(predicted_value)

        return np.array(predictions)

    def _score(self, dataset):
        """
        Calculate RMSE for a Dataset.
        """
        if not dataset.has_label():
            raise ValueError(
                "KNNRegressor requires labels to calculate score."
            )

        y_pred = self.predict(dataset)

        return rmse(
            dataset.y.astype(float),
            y_pred
        )