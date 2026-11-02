from abc import ABCMeta, abstractmethod 

import numpy as np
from copy import deepcopy


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


import numpy as np
from copy import deepcopy


class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None):
        n_features = self._input_shape[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        self.w_opt = deepcopy(optimizer)
        self.b_opt = deepcopy(optimizer)

        return self

    def parameters(self):
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = self.input @ self.weights + self.biases

        return self.output

    def backward_propagation(self, output_error):
        if self.w_opt is None or self.b_opt is None:
            raise ValueError(
                "Inicializa a camada com um optimizer antes do backward."
            )

        # Calcular antes de atualizar os pesos
        input_error = output_error @ self.weights.T

        weights_error = self.input.T @ output_error
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error

    def output_shape(self):
        return (self.n_units,)


class NoUpdate:

    def update(self, w, grad):
        self.last_gradient = grad.copy()
        return w


if __name__ == "__main__":
    np.random.seed(0)

    # Testes originais do forward
    layer = DenseLayer(4, (3,)).initialize(NoUpdate())

    X = np.random.rand(10, 3)
    out = layer.forward_propagation(X)

    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4
    assert layer.output_shape() == (4,)

    print("Todos os testes do forward passaram!")
    print("Nome da camada:", layer.layer_name())
    print("Forma da entrada por amostra:", layer.input_shape())
    print("Forma dos pesos:", layer.weights.shape)
    print("Forma dos biases:", layer.biases.shape)
    print("Forma da saída completa:", out.shape)
    print("Forma da saída por amostra:", layer.output_shape())
    print("Número de parâmetros:", layer.parameters())

    # Testes do backward e gradient check
    np.random.seed(0)

    X = np.random.rand(4, 3)
    Y = np.random.rand(4, 2)

    layer = DenseLayer(2, (3,)).initialize(NoUpdate())

    def mse(Y, P):
        return np.mean((Y - P) ** 2)

    def mse_derivative(Y, P):
        return 2 * (P - Y) / Y.size

    out = layer.forward_propagation(X, training=True)

    initial_weights = layer.weights.copy()
    initial_biases = layer.biases.copy()

    output_error = mse_derivative(Y, out)
    input_error = layer.backward_propagation(output_error)

    analytic = layer.w_opt.last_gradient.copy()
    bias_gradient = layer.b_opt.last_gradient.copy()

    numeric = np.zeros_like(layer.weights)
    eps = 1e-6

    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            original = layer.weights[i, j]

            layer.weights[i, j] = original + eps
            lp = mse(Y, layer.forward_propagation(X, training=True))

            layer.weights[i, j] = original - eps
            lm = mse(Y, layer.forward_propagation(X, training=True))

            layer.weights[i, j] = original
            numeric[i, j] = (lp - lm) / (2 * eps)

    max_error = np.max(np.abs(numeric - analytic))

    assert input_error.shape == X.shape
    assert analytic.shape == layer.weights.shape
    assert bias_gradient.shape == layer.biases.shape

    assert np.array_equal(layer.weights, initial_weights)
    assert np.array_equal(layer.biases, initial_biases)

    assert np.allclose(analytic, numeric, atol=1e-7, rtol=1e-5)
    assert np.allclose(
        input_error,
        output_error @ initial_weights.T
    )
    assert np.allclose(
        bias_gradient,
        np.sum(output_error, axis=0, keepdims=True)
    )

    print("\nTodos os testes do backward passaram!")
    print("Forma do gradiente da saída:", output_error.shape)
    print("Forma do gradiente da entrada:", input_error.shape)
    print("Forma do gradiente dos pesos:", analytic.shape)
    print("Forma do gradiente dos biases:", bias_gradient.shape)
    print("Loss (MSE):", mse(Y, out))

    print("\nGradiente analítico dos pesos:")
    print(analytic)

    print("\nGradiente numérico dos pesos:")
    print(numeric)

    print("\nGradiente dos biases:")
    print(bias_gradient)

    print("\nErro máximo do gradient check:", max_error)
    print(
        "Pesos mantidos pelo NoUpdate:",
        np.array_equal(layer.weights, initial_weights)
    )
    print(
        "Biases mantidos pelo NoUpdate:",
        np.array_equal(layer.biases, initial_biases)
    )

class Dropout(Layer):

    def __init__(self, probability=0.5, input_shape=None):
        if not 0 <= probability < 1:
            raise ValueError(
                "A probabilidade de dropout deve estar entre 0 e 1 "
                "(sem incluir 1)."
            )

        self.probability = probability
        self._input_shape = input_shape

        self.mask = None
        self.input = None
        self.output = None
        self.training = False

    def forward_propagation(self, input, training=False):
        self.input = input
        self._input_shape = input.shape[1:]
        self.training = training

        if training:
            scaling_factor = 1 / (1 - self.probability)

            self.mask = np.random.binomial(
                n=1,
                p=1 - self.probability,
                size=input.shape
            )

            self.output = input * self.mask * scaling_factor

        else:
            self.mask = None
            self.output = input

        return self.output

    def backward_propagation(self, output_error):
        if self.input is None:
            raise ValueError(
                "Executa forward_propagation antes do backward."
            )

        if self.training:
            scaling_factor = 1 / (1 - self.probability)
            return output_error * self.mask * scaling_factor

        return output_error

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0

if __name__ == "__main__":
    np.random.seed(0)

    X = np.random.rand(4, 5)
    layer = Dropout(probability=0.5)

    # Forward em modo de treino
    out_training = layer.forward_propagation(X, training=True)

    # Guardar a máscara antes de testar o modo de inferência
    training_mask = layer.mask.copy()
    scaling_factor = 1 / (1 - layer.probability)

    # Backward em modo de treino
    output_error = np.ones_like(X)
    input_error = layer.backward_propagation(output_error)

    assert out_training.shape == X.shape
    assert input_error.shape == X.shape
    assert layer.output_shape() == (5,)
    assert layer.parameters() == 0

    assert np.allclose(
        out_training,
        X * training_mask * scaling_factor
    )

    assert np.allclose(
        input_error,
        output_error * training_mask * scaling_factor
    )

    assert np.all(out_training[training_mask == 0] == 0)
    assert np.all(input_error[training_mask == 0] == 0)

    print("Todos os testes do Dropout em treino passaram!")
    print("Nome da camada:", layer.layer_name())
    print("Probabilidade de dropout:", layer.probability)
    print("Fator de escala:", scaling_factor)
    print("Forma da entrada por amostra:", layer.input_shape())
    print("Forma da saída completa:", out_training.shape)
    print("Forma da saída por amostra:", layer.output_shape())
    print("Número de parâmetros:", layer.parameters())

    print("\nEntrada:")
    print(X)

    print("\nMáscara de treino (0 = remover; 1 = manter):")
    print(training_mask)

    print("\nSaída em modo de treino:")
    print(out_training)

    print("\nGradiente recebido:")
    print(output_error)

    print("\nGradiente devolvido:")
    print(input_error)

    print("\nNúmero de elementos removidos:", np.sum(training_mask == 0))
    print(
        "Percentagem de elementos removidos neste teste:",
        f"{np.mean(training_mask == 0) * 100:.1f}%"
    )

    # Forward em modo de inferência
    out_inference = layer.forward_propagation(X, training=False)

    assert np.array_equal(out_inference, X)

    print("\nSaída em modo de inferência:")
    print(out_inference)
    print("Inferência mantém a entrada:", np.array_equal(out_inference, X))

    # Dropout com probabilidade zero
    layer_zero = Dropout(probability=0.0)
    out_zero = layer_zero.forward_propagation(X, training=True)

    assert np.array_equal(out_zero, X)

    print("\nDropout com probabilidade 0 mantém a entrada:",
          np.array_equal(out_zero, X))
    print("\nTodos os testes do Dropout passaram!")