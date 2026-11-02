import numpy as np
from abc import ABCMeta, abstractmethod
from layers import Layer



class ActivationLayer(Layer):

    def __init__(self):
        self.input = None
        self.output = None
        self._input_shape = None

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def forward_propagation(self, input, training=False):
        # Guardar a entrada da camada
        self.input = input

        # Aplicar a função de ativação
        self.output = self.activation_function(input)

        return self.output

    def backward_propagation(self, output_error):
        # Multiplicação elemento a elemento
        return output_error * self.derivative(self.input)

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        # Em zero, usamos a convenção de derivada igual a zero.
        return (input > 0).astype(float)


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        s = self.activation_function(input)
        return s * (1 - s)


class TanhActivation(ActivationLayer):

    def activation_function(self, input):
        return np.tanh(input)

    def derivative(self, input):
        t = self.activation_function(input)
        return 1 - t ** 2


class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):
        # Versão estável: subtrair o máximo de cada amostra.
        shifted_input = input - np.max(
            input,
            axis=-1,
            keepdims=True
        )

        exponentials = np.exp(shifted_input)

        return exponentials / np.sum(
            exponentials,
            axis=-1,
            keepdims=True
        )

    def derivative(self, input):
        # A derivada da softmax é um Jacobiano.
        # Para input (batch_size, n_classes), devolve
        # (batch_size, n_classes, n_classes).
        s = self.activation_function(input)

        n_classes = s.shape[-1]

        diagonal = s[..., :, None] * np.eye(n_classes)

        outer_product = (
            s[..., :, None] * s[..., None, :]
        )

        return diagonal - outer_product

    def backward_propagation(self, output_error):
        # A softmax precisa de um backward próprio.
        # Calcula o produto pelo Jacobiano sem o construir.
        s = self.output

        weighted_error = np.sum(
            output_error * s,
            axis=-1,
            keepdims=True
        )

        return s * (output_error - weighted_error)


if __name__ == "__main__":
    eps = 1e-6

    x = np.array([-2.5, -1.2, -0.3, 0.4, 1.1, 2.7])

    for act in [
        SigmoidActivation(),
        ReLUActivation(),
        TanhActivation()
    ]:
        numeric = (
            act.activation_function(x + eps)
            - act.activation_function(x - eps)
        ) / (2 * eps)

        analytic = act.derivative(x)

        error = np.abs(numeric - analytic).max()

        print(f"{act.layer_name()} — derivative error: {error}")

        assert error < 1e-6

        # Verificar o forward, o backward e os metadados
        act.set_input_shape((x.size,))

        output = act.forward_propagation(x, training=True)

        output_error = np.array([0.2, -0.4, 0.7, 0.1, -0.6, 0.3])
        input_error = act.backward_propagation(output_error)

        assert output.shape == x.shape
        assert input_error.shape == x.shape
        assert act.output_shape() == (x.size,)
        assert act.parameters() == 0

        np.testing.assert_allclose(
            input_error,
            output_error * analytic
        )

    # ==========================================
    # Softmax: estabilidade e probabilidades
    # ==========================================

    softmax = SoftmaxActivation()
    softmax.set_input_shape((3,))

    X = np.array([
        [1000.0, 1001.0, 1002.0],
        [-1000.0, -999.0, -998.0]
    ])

    probabilities = softmax.forward_propagation(
        X,
        training=True
    )

    assert probabilities.shape == X.shape
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert softmax.output_shape() == (3,)
    assert softmax.parameters() == 0

    np.testing.assert_allclose(
        probabilities.sum(axis=-1),
        np.ones(X.shape[0])
    )

    print("\nSoftmax probabilities:")
    print(probabilities)

    print("\nProbability sums per sample:")
    print(probabilities.sum(axis=-1))



    x_softmax = np.array([-0.5, 0.2, 1.0])

    analytic_jacobian = softmax.derivative(x_softmax)
    numeric_jacobian = np.zeros_like(analytic_jacobian)

    for j in range(x_softmax.size):
        x_plus = x_softmax.copy()
        x_minus = x_softmax.copy()

        x_plus[j] += eps
        x_minus[j] -= eps

        numeric_jacobian[:, j] = (
            softmax.activation_function(x_plus)
            - softmax.activation_function(x_minus)
        ) / (2 * eps)

    jacobian_error = np.abs(
        numeric_jacobian - analytic_jacobian
    ).max()

    print(f"\nSoftmax Jacobian error: {jacobian_error}")

    assert jacobian_error < 1e-6

    X_check = np.array([
        [-0.5, 0.2, 1.0],
        [0.8, -0.3, 0.4]
    ])

    output_error = np.array([
        [0.2, -0.4, 0.7],
        [-0.1, 0.6, 0.3]
    ])

    softmax.forward_propagation(X_check, training=True)
    analytic = softmax.backward_propagation(output_error)

    numeric = np.zeros_like(X_check)


    for index in np.ndindex(X_check.shape):
        X_plus = X_check.copy()
        X_minus = X_check.copy()

        X_plus[index] += eps
        X_minus[index] -= eps

        loss_plus = np.sum(
            softmax.activation_function(X_plus) * output_error
        )

        loss_minus = np.sum(
            softmax.activation_function(X_minus) * output_error
        )

        numeric[index] = (
            loss_plus - loss_minus
        ) / (2 * eps)

    backward_error = np.abs(numeric - analytic).max()

    print(f"Softmax backward error: {backward_error}")

    assert backward_error < 1e-6

    print("\nAll activation tests passed!")