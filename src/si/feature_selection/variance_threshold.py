import numpy as np

from ..base.transformer import Transformer
from ..data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Select features with a variance greater than a threshold.
    """

    def __init__(self, threshold=0.0):
        super().__init__()

        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset):
        """
        Calculate the variance of each feature.
        """
        self.variance = np.var(dataset.X, axis=0)

        return self

    def _transform(self, dataset):
        """
        Select features with variance greater than the threshold.
        """
        mask = self.variance > self.threshold

        if not np.any(mask):
            raise ValueError(
                "No feature has variance greater than the threshold."
            )

        X = dataset.X[:, mask]

        features = [
            feature
            for feature, selected in zip(dataset.features, mask)
            if selected
        ]

        return Dataset(
            X=X,
            y=dataset.y,
            features=features,
            label=dataset.label
        )