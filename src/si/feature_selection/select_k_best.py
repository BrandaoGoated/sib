import numpy as np

from ..base.transformer import Transformer
from ..data.dataset import Dataset
from ..statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Select the k features with the highest score.
    """

    def __init__(self, score_func=f_classification, k=10):
        super().__init__()

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset):
        """
        Calculate F-values and p-values using score_func.
        """
        self.F, self.p = self.score_func(dataset)

        return self

    def _transform(self, dataset):
        """
        Select the k features with the largest F-values.
        """
        if self.k > dataset.X.shape[1]:
            raise ValueError(
                "k cannot be greater than the number of features."
            )

        selected_indices = np.argsort(self.F)[-self.k:]

        X = dataset.X[:, selected_indices]

        features = [
            dataset.features[index]
            for index in selected_indices
        ]

        return Dataset(
            X=X,
            y=dataset.y,
            features=features,
            label=dataset.label
        )