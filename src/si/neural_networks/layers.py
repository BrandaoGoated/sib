from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training=False):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self):
        # Número de características de entrada
        n_features = self._input_shape[0]

        # Pesos aleatórios em [-0.5, 0.5[
        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        # Um bias por neurónio, inicialmente a zero
        self.biases = np.zeros((1, self.n_units))

        return self

    def parameters(self):
        # Número total de pesos e biases
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):
        # Guardar a entrada da camada
        self.input = input

        # Calcular a saída: Y = XW + b
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward_propagation(self, output_error):
        raise NotImplementedError(
            "A retropropagação da DenseLayer ainda não foi implementada."
        )


    def output_shape(self):
        # Forma da saída de uma amostra
        return (self.n_units,)


if __name__ == "__main__":
    # Camada com 3 características de entrada e 4 neurónios
    layer = DenseLayer(4, (3,)).initialize()

    # 10 amostras, cada uma com 3 características
    X = np.random.rand(10, 3)

    # Calcular a saída
    out = layer.forward_propagation(X)

    # Testes do exercício
    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4
    assert layer.output_shape() == (4,)

    print("Todos os testes passaram!")
    print("Nome da camada:", layer.layer_name())
    print("Forma da entrada por amostra:", layer.input_shape())
    print("Forma dos pesos:", layer.weights.shape)
    print("Forma dos biases:", layer.biases.shape)
    print("Forma da saída completa:", out.shape)
    print("Forma da saída por amostra:", layer.output_shape())
    print("Número de parâmetros:", layer.parameters())