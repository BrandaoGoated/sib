import numpy as np

from .losses import MeanSquaredError
from .optimizer import SGD


class NeuralNetwork:

    def __init__(
        self,
        epochs=100,
        batch_size=16,
        optimizer=SGD,
        learning_rate=0.01,
        verbose=False,
        loss=MeanSquaredError,
        metric=None,
        **kwargs
    ):
        """Inicializa a rede neuronal e os parâmetros de treino."""
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                "epochs deve ser um inteiro maior que zero."
            )

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size deve ser um inteiro maior que zero."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate deve ser maior que zero."
            )

        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss() if isinstance(loss, type) else loss
        self.metric = metric
        self.kwargs = kwargs

        self.layers = []
        self.history = {
            "loss": [],
            "metric": []
        }

    def add(self, layer):
        """Adiciona uma camada e inicializa os seus parâmetros."""
        if self.layers:
            layer.set_input_shape(
                self.layers[-1].output_shape()
            )

        elif layer.input_shape() is None:
            raise ValueError(
                "A primeira camada deve ter input_shape definido."
            )

        if callable(getattr(layer, "initialize", None)):
            optimizer = self.optimizer(
                learning_rate=self.learning_rate,
                **self.kwargs
            )

            layer.initialize(optimizer)

        self.layers.append(layer)

        return self

    def _forward(self, X, training=False):
        """Calcula a saída da rede por propagação direta."""
        output = X

        for layer in self.layers:
            output = layer.forward_propagation(
                output,
                training=training
            )

        return output

    def _align_targets(self, y, predictions):
        """Ajusta a forma dos rótulos à das previsões sem alterar a codificação."""
        y = np.asarray(y)

        if y.shape == predictions.shape:
            return y

        # Converter (n,) em (n, 1).
        if (
            predictions.ndim == 2
            and predictions.shape[1] == 1
            and y.ndim == 1
            and y.shape[0] == predictions.shape[0]
        ):
            return y.reshape(-1, 1)

        # Converter (n, 1) em (n,).
        if (
            predictions.ndim == 1
            and y.ndim == 2
            and y.shape[1] == 1
            and y.shape[0] == predictions.shape[0]
        ):
            return y.ravel()

        raise ValueError(
            f"A forma dos rótulos {y.shape} não corresponde "
            f"à forma das previsões {predictions.shape}. "
            "Para classificação multiclasse com "
            "CategoricalCrossEntropy, usa rótulos one-hot."
        )

    def fit(self, dataset):
        """Treina a rede e guarda a perda e a métrica de cada época."""
        if not self.layers:
            raise ValueError(
                "Adiciona camadas antes de treinar a rede."
            )

        if not dataset.has_label():
            raise ValueError(
                "O dataset de treino deve ter rótulos."
            )

        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y)

        n_samples = X.shape[0]

        if n_samples == 0:
            raise ValueError(
                "O dataset de treino não pode estar vazio."
            )

        if len(y) != n_samples:
            raise ValueError(
                "X e y devem ter o mesmo número de amostras."
            )

        if X.shape[1:] != self.layers[0].input_shape():
            raise ValueError(
                "A forma das features não corresponde "
                "à entrada da primeira camada."
            )

        self.history = {
            "loss": [],
            "metric": []
        }

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[
                    start:start + self.batch_size
                ]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                output = self._forward(
                    X_batch,
                    training=True
                )

                y_batch = self._align_targets(
                    y_batch,
                    output
                )

                error = self.loss.derivative(
                    y_batch,
                    output
                )

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

            predictions = self.predict(dataset)

            y_epoch = self._align_targets(
                y,
                predictions
            )

            epoch_loss = float(
                self.loss.loss(y_epoch, predictions)
            )

            self.history["loss"].append(epoch_loss)

            if self.metric is not None:
                epoch_metric = self.metric(
                    y_epoch,
                    predictions
                )

                self.history["metric"].append(
                    epoch_metric
                )

            if self.verbose:
                message = (
                    f"Epoch {epoch + 1}/{self.epochs}"
                    f" - loss: {epoch_loss:.6f}"
                )

                if self.metric is not None:
                    message += (
                        f" - metric: {epoch_metric:.6f}"
                    )

                print(message)

        return self

    def predict(self, dataset):
        """Calcula as previsões da rede em modo de inferência."""
        if not self.layers:
            raise ValueError(
                "Adiciona camadas antes de fazer previsões."
            )

        X = np.asarray(dataset.X, dtype=float)

        if X.shape[1:] != self.layers[0].input_shape():
            raise ValueError(
                "A forma das features não corresponde "
                "à entrada da primeira camada."
            )

        output = self._forward(X, training=False)

        if output.ndim == 2 and output.shape[1] == 1:
            return output.ravel()

        return output

    def score(self, dataset):
        """Calcula a métrica de avaliação no dataset fornecido."""
        if self.metric is None:
            raise ValueError(
                "É necessário definir uma métrica."
            )

        if not dataset.has_label():
            raise ValueError(
                "O dataset de avaliação deve ter rótulos."
            )

        if len(dataset.X) == 0:
            raise ValueError(
                "O dataset de avaliação não pode estar vazio."
            )

        predictions = self.predict(dataset)

        y_true = self._align_targets(
            dataset.y,
            predictions
        )

        return self.metric(y_true, predictions)

        