from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_true, y_pred):
        """Calcula a perda entre os valores reais e previstos."""
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """Calcula o gradiente da perda em relação às previsões."""
        pass


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        """Calcula a média dos quadrados das diferenças entre os valores reais e previstos."""
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_true, y_pred):
        """Calcula o gradiente do erro quadrático médio em relação às previsões."""
        return 2 * (y_pred - y_true) / y_pred.size


class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        """Calcula a entropia cruzada binária entre os rótulos reais e as probabilidades previstas."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    def derivative(self, y_true, y_pred):
        """Calcula o gradiente da entropia cruzada binária em relação às previsões."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


class CategoricalCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        """Calcula a entropia cruzada categórica entre os rótulos one-hot e as probabilidades previstas."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(y_true * np.log(y_pred))

    def derivative(self, y_true, y_pred):
        """Calcula o gradiente da entropia cruzada categórica em relação às previsões."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -y_true / y_pred