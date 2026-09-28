import numpy as np

from ..base.model import Model
from ..metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression solved with the closed-form Least Squares equation.
    """

    def __init__(self, l2_penalty=1.0, scale=True):
        super().__init__()

        if l2_penalty < 0:
            raise ValueError(
                "l2_penalty must be greater than or equal to 0."
            )

        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _scale_fit(self, X):
        """
        Estimate scaling values and scale X.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1

        return (X - self.mean) / self.std

    def _scale_predict(self, X):
        """
        Scale new X using fit-time mean and standard deviation.
        """
        return (X - self.mean) / self.std

    def _fit(self, dataset):
        """
        Estimate model parameters using Ridge closed-form solution.
        """
        if not dataset.has_label():
            raise ValueError(
                "RidgeRegressionLeastSquares requires labels."
            )

        X = dataset.X.astype(float)
        y = dataset.y.astype(float)

        if self.scale:
            X = self._scale_fit(X)
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])

        n_samples, n_features = X.shape

        X_with_intercept = np.c_[
            np.ones(n_samples),
            X
        ]

        penalty_matrix = self.l2_penalty * np.eye(
            n_features + 1
        )

        penalty_matrix[0, 0] = 0

        parameters = np.linalg.inv(
            X_with_intercept.T.dot(X_with_intercept)
            + penalty_matrix
        ).dot(
            X_with_intercept.T
        ).dot(y)

        self.theta_zero = parameters[0]
        self.theta = parameters[1:]

        return self

    def _predict(self, dataset):
        """
        Predict target values for a Dataset.
        """
        X = dataset.X.astype(float)

        if self.scale:
            X = self._scale_predict(X)

        n_samples = X.shape[0]

        X_with_intercept = np.c_[
            np.ones(n_samples),
            X
        ]

        parameters = np.r_[
            self.theta_zero,
            self.theta
        ]

        return X_with_intercept.dot(parameters)

    def _score(self, dataset):
        """
        Calculate Mean Squared Error for a Dataset.
        """
        if not dataset.has_label():
            raise ValueError(
                "RidgeRegressionLeastSquares requires labels."
            )

        y_pred = self.predict(dataset)

        return mse(
            dataset.y.astype(float),
            y_pred
        )