import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    Logistic Regression trained with gradient descent and L2 regularization.
    """

    def __init__(
        self,
        l2_penalty=0.0,
        alpha=0.01,
        max_iter=1000,
        patience=100,
        scale=True
    ):
        super().__init__()

        if l2_penalty < 0:
            raise ValueError("l2_penalty must be greater than or equal to 0.")
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0.")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")
        if patience <= 0:
            raise ValueError("patience must be greater than 0.")

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
        """Estimate feature means and standard deviations, then scale X."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        return (X - self.mean) / self.std

    def _scale_predict(self, X):
        """Scale features using the values estimated during fitting."""
        return (X - self.mean) / self.std

    def _fit(self, dataset):
        """Fit the model using gradient descent."""
        if not dataset.has_label():
            raise ValueError("LogisticRegression requires labels.")

        X = dataset.X.astype(float)
        y = dataset.y.astype(float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError(
                "LogisticRegression requires labels encoded as 0 and 1."
            )

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
            probabilities = sigmoid_function(
                X.dot(self.theta) + self.theta_zero
            )
            error = probabilities - y

            # Gradient of binary cross-entropy plus L2 regularization.
            gradient_theta = (
                X.T.dot(error) / n_samples
                + (self.l2_penalty / n_samples) * self.theta
            )
            gradient_theta_zero = np.mean(error)

            self.theta -= self.alpha * gradient_theta
            self.theta_zero -= self.alpha * gradient_theta_zero

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
        """Predict binary labels using a probability threshold of 0.5."""
        if self.theta is None:
            raise ValueError(
                "LogisticRegression must be fitted before prediction."
            )

        X = dataset.X.astype(float)

        if self.scale:
            X = self._scale_predict(X)

        probabilities = sigmoid_function(
            X.dot(self.theta) + self.theta_zero
        )

        return (probabilities >= 0.5).astype(int)

    def _score(self, dataset):
        """Calculate accuracy between the actual and predicted labels."""
        if not dataset.has_label():
            raise ValueError(
                "LogisticRegression requires labels to calculate score."
            )

        y_true = dataset.y.astype(int).reshape(-1)
        y_pred = self.predict(dataset).astype(int).reshape(-1)

        return accuracy(y_true, y_pred)

    def cost(self, dataset):
        """
        Calculate binary cross-entropy plus the slide's L2 penalty.

        The intercept theta_zero is not included in the regularization term.
        """
        if not dataset.has_label():
            raise ValueError(
                "LogisticRegression requires labels to calculate cost."
            )
        if self.theta is None:
            raise ValueError(
                "LogisticRegression must be fitted before calculating cost."
            )

        X = dataset.X.astype(float)
        y = dataset.y.astype(float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError(
                "LogisticRegression requires labels encoded as 0 and 1."
            )

        if self.scale:
            X = self._scale_predict(X)

        logits = X.dot(self.theta) + self.theta_zero

        # Stable equivalent of:
        # -mean(y * log(p) + (1 - y) * log(1 - p))
        cross_entropy = np.mean(
            np.logaddexp(0.0, logits) - y * logits
        )

        n_samples = X.shape[0]
        regularization = (
            self.l2_penalty / (2.0 * n_samples)
        ) * np.sum(self.theta ** 2)

        return cross_entropy + regularization