from typing import NamedTuple, Sequence

from numpy import ndarray


class Perceptron(NamedTuple):
    weights: ndarray[float]
    bias: float

    def predict(self, inputs: Sequence[float]) -> bool:
        value = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        if value > 0:
            return True
        return False
