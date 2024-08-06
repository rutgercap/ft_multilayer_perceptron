from pathlib import Path

import pytest
from pytest import fixture, raises

from src.network import (
    HiddenLayer,
    MultiLayerPerceptron,
    Neuron,
    Perceptron,
    SoftmaxLayer,
)


@fixture
def perceptron() -> Perceptron:
    weights = [1, 1]
    bias = 0
    return Perceptron(weights, bias)


def test_perceptron_returns_false_if_values_are_0() -> None:
    inputs = [0, 0]
    weights = [1, 1]
    bias = 0
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == False


def test_perceptron_raises_error_if_input_does_not_match_weights() -> None:
    inputs = [0]
    weights = [1, 1]
    bias = 0
    perceptron = Perceptron(weights, bias)

    with raises(ValueError):
        perceptron.predict(inputs)


def test_perceptron_returns_true_if_values_are_1() -> None:
    inputs = [1, 1]
    weights = [1, 1]
    bias = 0
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == True


def test_perceptron_ignores_inputs_with_0_weight() -> None:
    inputs = [1, 0]
    weights = [0, 1]
    bias = 0
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == False


def test_perceptron_returns_true_if_high_bias() -> None:
    inputs = [0, 0]
    weights = [0, 0]
    bias = 1
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == True


def test_perceptron_returns_false_if_low_bias() -> None:
    inputs = [0, 0]
    weights = [0, 0]
    bias = -1
    perceptron = Perceptron(weights, bias)

    result = perceptron.predict(inputs)

    assert result == False


def test_can_create_network() -> None:
    layer = HiddenLayer(size=2)
    output_layer = SoftmaxLayer()
    network = MultiLayerPerceptron(
        input_size=2, hidden_layers=[layer], output_layer=output_layer
    )
    network.initialize()

    result = network.predict([1, 2])

    assert len(result) == 2


def test_network_raises_exception_if_incorrect_input_given() -> None:
    layer = HiddenLayer(size=1)
    output_layer = SoftmaxLayer()
    network = MultiLayerPerceptron(
        input_size=1, hidden_layers=[layer], output_layer=output_layer
    )
    network.initialize()

    with raises(ValueError):
        network.predict([1, 1])


def test_softmax_returns_correct_output() -> None:
    layer = SoftmaxLayer()

    result = layer.forward([1, 1])

    assert result == [0.5, 0.5]


def test_softmax_works_with_more_complicated_input() -> None:
    layer = SoftmaxLayer()

    result = layer.forward([1, 2, 3])

    assert [round(x, 5) for x in result] == [
        round(x, 5)
        for x in [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]
    ]


def test_hidden_layer_forward_raises_error_if_not_initialized() -> None:
    layer = HiddenLayer(size=1)

    with raises(ValueError):
        layer.forward([1, 1])


def test_hidden_layer_works_with_neurons() -> None:
    layer = HiddenLayer.from_neurons(
        [Neuron([0.0, 0.0], 0.0), Neuron([0.0, 0.0], 1.0), Neuron([1.0, 1.0], 0.0)]
    )

    result = layer.forward([1, 1])

    assert result == [0, 1, 2]


def test_can_create_network_with_premade_neurons() -> None:
    layer = HiddenLayer.from_neurons(
        [Neuron([0.0, 0.0], 0.0), Neuron([0.0, 0.0], 1.0), Neuron([1.0, 1.0], 0.0)]
    )
    output_layer = SoftmaxLayer()
    network = MultiLayerPerceptron(
        input_size=2, hidden_layers=[layer], output_layer=output_layer
    )

    result = network.predict([1, 1])

    assert [round(x, 5) for x in result] == [
        round(x, 5)
        for x in [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]
    ]
