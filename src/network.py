import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy import ndarray


def sigmoid(x: ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: ndarray):
    return sigmoid(x) * (1 - sigmoid(x))


def binary_cross_entropy(y_true: ndarray, y_pred: ndarray):
    return -np.mean(
        y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
    )


class HiddenLayer:
    weights: ndarray
    biases: ndarray
    z: ndarray
    output: ndarray
    inputs: ndarray

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def weights_json(self):
        return {
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

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

    def backward(
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
        output_error = self.output_layer.backward(hidden_output, y, learning_rate)
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(
        self,
        data: tuple[ndarray, ndarray],
        learning_rate: float,
        epochs: int,
        silent=False,
        validation_data: Optional[tuple[ndarray, ndarray]] = None,
    ) -> Sequence[float]:
        X, y = data

        loss_history = []
        for i in range(epochs):
            training_output = self.forward(X)
            training_loss = binary_cross_entropy(y, training_output)
            loss_history.append(training_loss)
            if validation_data:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)
                val_loss = binary_cross_entropy(y_val, val_output)
                if not silent:
                    print(
                        f"Epoch {i + 1} / {epochs}, Loss: {training_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
            elif not silent:
                print(f"Epoch {i + 1} / {epochs}, Loss: {training_loss:.4f}")
            self.backward(X, y, learning_rate)
        return loss_history

    def save(self, path: Path) -> None:
        weights = {
            f"hidden_layer_{i}": layer.weights_json()
            for i, layer in enumerate(self.layers)
        }
        weights["output_layer"] = self.output_layer.weights_json()
        json.dump(weights, open(path, "w"))

    def load(self, path: Path) -> None:
        with open(path) as f:
            json_layers = json.load(f)
            for i, layer in enumerate(self.layers):
                layer.weights = np.array(json_layers[f"hidden_layer_{i}"]["weights"])
                layer.biases = np.array(json_layers[f"hidden_layer_{i}"]["biases"])
            self.output_layer.weights = np.array(json_layers["output_layer"]["weights"])
            self.output_layer.biases = np.array(json_layers["output_layer"]["biases"])
