import numpy as np

from ..base.model import Model
from ..metrics.accuracy import accuracy


def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two samples.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier(Model):
    """
    K-Nearest Neighbors classifier.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        """
        Initialize the classifier.
        """
        super().__init__()

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset):
        """
        Store the training Dataset.
        """
        if not dataset.has_label():
            raise ValueError(
                "KNNClassifier requires a Dataset with labels."
            )

        if self.k > dataset.X.shape[0]:
            raise ValueError(
                "k cannot be greater than the number of training samples."
            )

        self.dataset = dataset

        return self

    def _predict(self, dataset):
        """
        Predict the class for every sample in a Dataset.
        """
        predictions = []

        for sample in dataset.X:
            distances = np.array([
                self.distance(sample, train_sample)
                for train_sample in self.dataset.X
            ])

            nearest_indices = np.argsort(distances)[:self.k]

            nearest_classes = self.dataset.y[nearest_indices]

            classes, counts = np.unique(
                nearest_classes,
                return_counts=True
            )

            predicted_class = classes[np.argmax(counts)]

            predictions.append(predicted_class)

        return np.array(predictions)

    def _score(self, dataset):
        """
        Calculate classification accuracy on a Dataset.
        """
        if not dataset.has_label():
            raise ValueError(
                "KNNClassifier requires a Dataset with labels to score."
            )

        y_pred = self.predict(dataset)

        return accuracy(dataset.y, y_pred)