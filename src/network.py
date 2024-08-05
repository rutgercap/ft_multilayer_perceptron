import math
from pathlib import Path
from typing import Sequence

from numpy import dot
from numpy.random import random


class Perceptron:
    weights: Sequence[float]
    bias: float

    def __init__(self, weights: Sequence[float], bias: float):
        assert isinstance(weights, list) and len(weights) > 0
        assert all(
            isinstance(weight, float) or isinstance(weight, int) for weight in weights
        )
        assert isinstance(bias, float) or isinstance(bias, int)
        self.weights = weights
        self.bias = bias
        self

    def predict(self, inputs: Sequence[float]) -> float:
        value = dot(self.weights, inputs) + self.bias
        return 1 if value > 0 else 0


class Neuron:
    weights: Sequence[float]
    bias: float

    def __init__(self, weights: Sequence[float], bias: float):
        assert isinstance(weights, list) and len(weights) > 0
        assert all(
            isinstance(weight, float) or isinstance(weight, int) for weight in weights
        )
        assert isinstance(bias, float) or isinstance(bias, int)
        self.weights = weights
        self.bias = bias

    def predict(self, inputs: Sequence[float]) -> float:
        value = dot(self.weights, inputs) + self.bias
        return value

    def to_json_str(self) -> str:
        return f'{{"weights": {self.weights}, "bias": {self.bias}}}'


class HiddenLayer:
    _neurons: list[Neuron]
    size: int

    def __init__(self, size: int):
        assert isinstance(size, int) and size > 0
        self.size = size
        self._neurons = []

    @classmethod
    def from_neurons(cls, neurons: list[Neuron]):
        layer = cls(len(neurons))
        layer._neurons = neurons
        layer.size = len(neurons)
        return layer

    def forward(self, inputs: Sequence[float]) -> Sequence[float]:
        if len(self._neurons) == 0:
            raise ValueError("Neurons uninitialized")
        return [neuron.predict(inputs) for neuron in self._neurons]

    def initialize(self, input_size: int):
        for _ in range(self.size):
            weights = [random() for _ in range(input_size)]
            bias = 0
            self._neurons.append(Neuron(weights, bias))

    def to_json_str(self) -> str:
        return f'{{"size": {self.size}, "neurons": [{", ".join([neuron.to_json_str() for neuron in self._neurons])}]}}'


class SoftmaxLayer:
    def forward(self, inputs: Sequence[float]) -> Sequence[float]:
        values = [math.exp(x) for x in inputs]
        values_sum = sum(values)
        return [x / values_sum for x in values]

    def to_json_str(self) -> str:
        return "softmax"


class MultiLayerPerceptron:
    input_size: int
    output_layer: SoftmaxLayer
    hidden_layers: Sequence[HiddenLayer]

    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[HiddenLayer],
        output_layer: SoftmaxLayer,
    ):
        assert isinstance(input_size, int) and input_size > 0
        assert isinstance(hidden_layers, list) and len(hidden_layers) > 0
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def initialize(self):
        input_size = self.input_size
        for layer in self.hidden_layers:
            layer.initialize(input_size)
            input_size = layer.size

    def predict(self, inputs: Sequence[float]) -> Sequence[float]:
        outputs = inputs
        for layer in self.hidden_layers:
            outputs = layer.forward(outputs)
        return self.output_layer.forward(outputs)

    def to_file(self, path: Path) -> None:
        with open(path, "w+") as f:
            f.write(self.to_json_str())

    def to_json_str(self) -> str:
        return f'{{"input_size": {self.input_size}, "hidden_layers": [{", ".join([layer.to_json_str() for layer in self.hidden_layers])}], "output_layer": {self.output_layer.to_json_str()}}}'

    @staticmethod
    def from_json_str(data: str) -> "MultiLayerPerceptron":
        raise NotImplementedError()

    @staticmethod
    def from_file(path: Path) -> "MultiLayerPerceptron":
        with open(path, "r") as f:
            data = f.read()
        return MultiLayerPerceptron.from_json_str(data)
