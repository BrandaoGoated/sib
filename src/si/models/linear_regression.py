import numpy as np

from ..base.model import Model
from ..metrics.mse import mse


class RidgeRegression(Model):
    """
    Ridge Regression trained with Gradient Descent.
    """

    def __init__(
        self,
        l2_penalty=1.0,
        alpha=0.01,
        max_iter=1000,
        patience=100,
        scale=True
    ):
        super().__init__()

        if l2_penalty < 0:
            raise ValueError(
                "l2_penalty must be greater than or equal to 0."
            )

        if alpha <= 0:
            raise ValueError(
                "alpha must be greater than 0."
            )

        if max_iter <= 0:
            raise ValueError(
                "max_iter must be greater than 0."
            )

        if patience <= 0:
            raise ValueError(
                "patience must be greater than 0."
            )

        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

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
        Scale new X with values estimated during fitting.
        """
        return (X - self.mean) / self.std

    def _fit(self, dataset):
        """
        Estimate Ridge Regression parameters using Gradient Descent.
        """
        if not dataset.has_label():
            raise ValueError(
                "RidgeRegression requires a Dataset with labels."
            )

        X = dataset.X.astype(float)
        y = dataset.y.astype(float)

        if self.scale:
            X = self._scale_fit(X)
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])

        n_samples, n_features = X.shape

        self.theta = np.zeros(n_features)
        self.theta_zero = 0.0
        self.cost_history = {}

        best_cost = np.inf
        iterations_without_improvement = 0

        for iteration in range(self.max_iter):
            y_pred = X.dot(self.theta) + self.theta_zero

            error = y_pred - y

            gradient_theta = (
                2 / n_samples
            ) * X.T.dot(error) + (
                2 * self.l2_penalty / n_samples
            ) * self.theta

            gradient_theta_zero = 2 * np.mean(error)

            self.theta = self.theta - self.alpha * gradient_theta
            self.theta_zero = (
                self.theta_zero - self.alpha * gradient_theta_zero
            )

            current_cost = self.cost(dataset)
            self.cost_history[iteration] = current_cost

            if current_cost < best_cost:
                best_cost = current_cost
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.patience:
                break

        return self

    def _predict(self, dataset):
        """
        Predict target values for a Dataset.
        """
        X = dataset.X.astype(float)

        if self.scale:
            X = self._scale_predict(X)

        return X.dot(self.theta) + self.theta_zero

    def _score(self, dataset):
        """
        Calculate the Mean Squared Error for a Dataset.
        """
        if not dataset.has_label():
            raise ValueError(
                "RidgeRegression requires labels to calculate score."
            )

        y_pred = self.predict(dataset)

        return mse(
            dataset.y.astype(float),
            y_pred
        )

    def cost(self, dataset):
        """
        Calculate Ridge cost: MSE plus L2 regularization.
        """
        if not dataset.has_label():
            raise ValueError(
                "RidgeRegression requires labels to calculate cost."
            )

        y_pred = self._predict(dataset)

        squared_error = np.mean(
            (dataset.y.astype(float) - y_pred) ** 2
        )

        regularization = self.l2_penalty * np.sum(
            self.theta ** 2
        )

        return squared_error + regularization