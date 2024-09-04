import json
import time
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy import array, ndarray


def sigmoid(x: ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def binary_cross_entropy(y_true: ndarray, y_pred: ndarray):
    return -np.mean(
        y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
    )


class TrainingOverview(NamedTuple):
    duration: float
    epochs: int
    learning_rate: float
    loss_history: Sequence[float]
    accuracy_history: Sequence[float]
    validation_accuracy_history: Optional[Sequence[float]] = None

    def __str__(self):
        return f"Epochs: {self.epochs}, Learning Rate: {self.learning_rate}, Loss: {self.loss}, Validation Loss: {self.validation_loss}"

    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, self.epochs + 1),
            self.loss_history,
            label="Training Loss",
            color="red",
        )
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, self.epochs + 1),
            self.accuracy_history,
            label="Training Accuracy",
            color="green",
        )

        if self.validation_accuracy_history:
            plt.plot(
                range(1, self.epochs + 1),
                self.validation_accuracy_history,
                label="Validation Accuracy",
                color="blue",
            )

        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()


class HiddenLayer:
    weights: ndarray
    biases: ndarray
    z: ndarray
    output: ndarray
    inputs: ndarray

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def weights_json(self) -> dict[str, list]:
        return {
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

    @classmethod
    def from_weights(cls, weights: ndarray, biases: ndarray) -> "HiddenLayer":
        layer = HiddenLayer(0, 0)
        layer.weights = weights
        layer.biases = biases
        return layer

    def forward(self, X: ndarray):
        self.inputs = X
        self.z = np.dot(X, self.weights) + self.biases
        self.output = self.activate(self.z)
        return self.output

    def predict(self, X: ndarray):
        z = np.dot(X, self.weights) + self.biases
        return self.activate(z)

    def activate(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def backward_output_layer(
        self, X: ndarray, y: ndarray, learning_rate: float
    ) -> ndarray:
        m = y.shape[0]
        error = self.output - y
        weights_gradient = np.dot(X.T, error) / m
        bias_gradient = np.sum(error, axis=0, keepdims=True) / m
        self.biases -= learning_rate * bias_gradient
        self.weights -= learning_rate * weights_gradient
        return np.dot(error, self.weights.T)

    def backward(self, error: ndarray, learning_rate: float) -> ndarray:
        sigmoid_deriv = sigmoid_derivative(self.z)
        error = error * sigmoid_deriv

        input_error = np.dot(error, self.weights.T)
        weights_gradient = np.dot(self.inputs.T, error) / self.inputs.shape[0]
        bias_gradient = np.sum(error, axis=0, keepdims=True) / self.inputs.shape[0]

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * bias_gradient

        return input_error


class SoftmaxLayer(HiddenLayer):
    def activate(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        rounded_output = np.round(output, decimals=4)
        return rounded_output

    def backward_output_layer(
        self, prev_layer_output: ndarray, y: ndarray, learning_rate: float
    ) -> ndarray:
        m = y.shape[0]
        error = self.output - y
        weights_gradient = np.dot(prev_layer_output.T, error) / m
        bias_gradient = np.sum(error, axis=0, keepdims=True) / m

        self.biases -= learning_rate * bias_gradient
        self.weights -= learning_rate * weights_gradient
        return np.dot(error, self.weights.T)


class MLP:
    input_size: int
    layers: Sequence[HiddenLayer]
    output_layer: HiddenLayer

    def __init__(
        self, input_size: int, hidden_layer_sizes: Sequence[int], output_size: int
    ):
        self.input_size = input_size
        layers: list[HiddenLayer] = []
        previous_layer_size = input_size
        for layer in hidden_layer_sizes:
            layers.append(HiddenLayer(previous_layer_size, layer))
            previous_layer_size = layer
        self.layers = layers
        self.output_layer = SoftmaxLayer(previous_layer_size, output_size)

    @classmethod
    def from_weights(
        cls,
        input_size: int,
        hidden_layer_weights: Sequence[tuple[ndarray, ndarray]],
        output_layer_weights: tuple[ndarray, ndarray],
        output_size: int,
    ) -> "MLP":
        mlp = MLP(
            input_size, [i for i in range(len(hidden_layer_weights))], output_size
        )
        mlp.output_layer.weights = output_layer_weights[0]
        mlp.output_layer.biases = output_layer_weights[1]
        for i, (weights, biases) in enumerate(hidden_layer_weights):
            mlp.layers[i].weights = weights
            mlp.layers[i].biases = biases
        return mlp

    def forward(self, X: ndarray) -> ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return self.output_layer.forward(X)

    def predict(self, X: ndarray) -> ndarray:
        for layer in self.layers:
            X = layer.predict(X)
        return self.output_layer.predict(X)

    def backward(self, X: ndarray, y: ndarray, learning_rate: float):
        hidden_output = X
        for layer in self.layers:
            hidden_output = layer.forward(hidden_output)
        output_error = self.output_layer.backward_output_layer(
            hidden_output, y, learning_rate
        )
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def accuracy(self, y_true: ndarray, y_pred: ndarray) -> float:
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    def train(
        self,
        data: tuple[ndarray, ndarray],
        learning_rate: float,
        epochs: int,
        batch_size: Optional[int] = None,
        silent=False,
        validation_data: Optional[tuple[ndarray, ndarray]] = None,
    ) -> TrainingOverview:
        X, y = data

        start_time = time.time()
        loss_history = []
        accuracy_history = []
        val_accuracy_history = []
        for i in range(epochs):
            if not batch_size:
                x_batch = X
                y_batch = y
            else:
                indices = np.random.choice(X.shape[0], batch_size, replace=False)
                x_batch = X[indices]
                y_batch = y[indices]
            training_output = self.forward(x_batch)
            training_loss = float(binary_cross_entropy(y_batch, training_output))
            loss_history.append(training_loss)
            accuracy_history.append(float(self.accuracy(y_batch, training_output)))
            if validation_data:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)
                val_accuracy_history.append(float(self.accuracy(y_val, val_output)))
                val_loss = binary_cross_entropy(y_val, val_output)
                if not silent:
                    print(
                        f"Epoch {i + 1} / {epochs}, Loss: {training_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
            elif not silent:
                print(f"Epoch {i + 1} / {epochs}, Loss: {training_loss:.4f}")
            self.backward(x_batch, y_batch, learning_rate)
        end_time = time.time()
        duration = end_time - start_time
        return TrainingOverview(
            duration,
            epochs,
            learning_rate,
            loss_history,
            accuracy_history,
            val_accuracy_history,
        )

    def save(self, path: Path) -> None:
        weights = {
            "input_size": self.input_size,
            "output_layer": self.output_layer.weights_json(),
            "layers": {i: layer.weights_json() for i, layer in enumerate(self.layers)},
        }
        json.dump(weights, open(path, "w"))

    @classmethod
    def from_file(cls, path: Path) -> "MLP":
        with open(path) as f:
            json_layers = json.load(f)
            input_size: int = json_layers["input_size"]
            output_layer = (
                array(json_layers["output_layer"]["weights"]),
                array(json_layers["output_layer"]["biases"]),
            )
            output_size = len(output_layer[1])
            layers: dict[int, dict[str, Sequence]] = json_layers["layers"]
            hidden_layers = [
                (
                    array(layers[i]["weights"]),
                    array(layers[i]["biases"]),
                )
                for i in sorted(layers.keys())
            ]
            return MLP.from_weights(
                input_size, hidden_layers, output_layer, output_size
            )
