from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    def __init__(self, learning_rate=0.01):
        """Inicializa o optimizer com a taxa de aprendizagem."""
        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser maior que zero.")

        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, grad_loss_w):
        """Calcula os parâmetros atualizados a partir dos gradientes."""
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0):
        """Inicializa o SGD com a taxa de aprendizagem e o momentum."""
        super().__init__(learning_rate)

        if not 0 <= momentum < 1:
            raise ValueError(
                "momentum deve estar entre 0 e 1, sem incluir 1."
            )

        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        """Calcula o gradiente acumulado e devolve os parâmetros atualizados."""
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w, dtype=float)

        self.retained_gradient = (
            self.momentum * self.retained_gradient
            + (1 - self.momentum) * grad_loss_w
        )

        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):

    def __init__(
        self,
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    ):
        """Inicializa o Adam e os parâmetros das médias móveis."""
        super().__init__(learning_rate)

        if not 0 <= beta_1 < 1:
            raise ValueError(
                "beta_1 deve estar entre 0 e 1, sem incluir 1."
            )

        if not 0 <= beta_2 < 1:
            raise ValueError(
                "beta_2 deve estar entre 0 e 1, sem incluir 1."
            )

        if epsilon <= 0:
            raise ValueError("epsilon deve ser maior que zero.")

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_loss_w):
        """Calcula as médias móveis corrigidas e devolve os parâmetros atualizados."""
        if self.m is None:
            self.m = np.zeros_like(w, dtype=float)
            self.v = np.zeros_like(w, dtype=float)

        self.t += 1

        self.m = (
            self.beta_1 * self.m
            + (1 - self.beta_1) * grad_loss_w
        )

        self.v = (
            self.beta_2 * self.v
            + (1 - self.beta_2) * grad_loss_w ** 2
        )

        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        return (
            w
            - self.learning_rate
            * m_hat
            / (np.sqrt(v_hat) + self.epsilon)
        )