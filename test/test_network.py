from pytest import fixture, raises

import pandas as pd
from numpy import array, all
from src.dataset import dataframe_to_numpy
from src.network import (
    SoftmaxLayer,
    MLP
)

def test_softmax_returns_correct_output() -> None:
    layer = SoftmaxLayer()

    result = layer.forward(array([1, 1]))

    assert all(result == [0.5, 0.5])


def test_softmax_works_with_more_complicated_input() -> None:
    layer = SoftmaxLayer()

    result = layer.forward(array([1, 2, 3]))

    assert [round(x, 5) for x in result] == [
        round(x, 5)
        for x in [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]
    ]


def test_can_create_correct_network_with_hidden_layers() -> None:
    output_size = 2
    network = MLP(input_size=2, hidden_layer_sizes=[3, 3], output_size=output_size)
    
    assert network.layers[0].weights.shape == (2, 3)
    assert network.layers[0].biases.shape == (1, 3)
    assert network.layers[1].weights.shape == (3, 3)
    assert network.layers[1].biases.shape == (1, 3)
    assert network.output_layer.weights.shape == (3, 2)
    assert network.output_layer.biases.shape == (1, 2)

    result = network.forward(array([1, 2]))

    assert len(result) == output_size


def test_can_create_correct_network_without_hidden_layers() -> None:
    output_size = 2
    network = MLP(input_size=2, hidden_layer_sizes=[], output_size=output_size)
    
    assert network.output_layer.weights.shape == (2, 2)
    assert network.output_layer.biases.shape == (1, 2)

    result = network.forward(array([1, 2]))

    assert len(result) == output_size


def test_network_raises_error_if_input_incorrect_dimension() -> None:
    with raises(ValueError):
        network = MLP(input_size=2, hidden_layer_sizes=[3, 3], output_size=2)
        network.forward(array([1, 2, 3]))

