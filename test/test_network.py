from numpy import ndarray

from src.network import Perceptron


def test_perceptron_returns_false_if_values_are_0() -> None:
    inputs = [0, 0]
    weights = ndarray([1, 1])
    bias = 0
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == False


def test_perceptron_returns_true_if_values_are_1() -> None:
    inputs = [1, 1]
    weights = ndarray([1, 1])
    bias = 0
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == True


def test_perceptron_ignores_inputs_with_0_weight() -> None:
    inputs = [1, 0]
    weights = ndarray([0, 1])
    bias = 0
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == False


def test_perceptron_returns_true_if_high_bias() -> None:
    inputs = [0, 0]
    weights = ndarray([0, 0])
    bias = 1
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == True
