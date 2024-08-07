import math
from typing import Any, Sequence

from numpy import dot, ndarray, array
import numpy as np

class SoftmaxLayer:
    def forward(self, inputs: ndarray) -> ndarray:
        values = [math.exp(x) for x in inputs]
        values_sum = sum(values)
        return array([x / values_sum for x in values])


def sigmoid(x: ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: ndarray):
    return sigmoid(x) * (1 - sigmoid(x))

def binary_cross_entropy(y_true: ndarray, y_pred: ndarray):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class HiddenLayer:
    def __init__(self, input_size: int, hidden_size: int):
        self.weights = np.random.randn(input_size, hidden_size)
        self.biases = np.zeros((1, hidden_size))

    def forward(self, X: ndarray):
        self.z = np.dot(X, self.weights) + self.biases
        self.output = sigmoid(self.z)
        return self.output

class MLP:
    input_size: int
    layers: Sequence[HiddenLayer]
    output_layer: HiddenLayer

    def __init__(self, input_size: int, hidden_layer_sizes: Sequence[int], output_size: int):
        self.input_size = input_size
        layers: list[HiddenLayer] = []
        previous_layer_size = input_size
        for layer in hidden_layer_sizes:
            layers.append(HiddenLayer(previous_layer_size, layer))
            previous_layer_size = layer
        self.layers = layers
        self.output_layer = HiddenLayer(previous_layer_size, output_size)


    def forward(self, X: ndarray):
        raise NotImplementedError()

    def backward(self, X: ndarray, y: ndarray, learning_rate: float):
        raise NotImplementedError()

    def train(self, X: ndarray, y: ndarray, learning_rate: float, epochs: int):
        raise NotImplementedError()
