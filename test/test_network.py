from pathlib import Path

from numpy import array
from pytest import raises

from src.dataset import dataframe_to_numpy
from src.network import MLP


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

    assert len(result[0]) == output_size


def test_can_create_correct_network_without_hidden_layers() -> None:
    output_size = 2
    network = MLP(input_size=2, hidden_layer_sizes=[], output_size=output_size)

    assert network.output_layer.weights.shape == (2, 2)
    assert network.output_layer.biases.shape == (1, 2)

    result = network.forward(array([1, 2]))

    assert len(result[0]) == output_size


def test_network_raises_error_if_input_incorrect_dimension() -> None:
    with raises(ValueError):
        network = MLP(input_size=2, hidden_layer_sizes=[3, 3], output_size=2)
        network.forward(array([1, 2, 3]))


def test_can_train_network() -> None:
    output_size = 2
    network = MLP(input_size=2, hidden_layer_sizes=[3, 4], output_size=output_size)
    X = array([[1, 2], [3, 4], [5, 6]])
    y = array([[1, 0], [0, 1], [1, 0]])
    learning_rate = 0.1
    epochs = 10

    result = network.forward([1, 2])
    network.train(X, y, learning_rate, epochs)

    assert len(result[0]) == output_size


def test_can_save_weights_to_file(tmpdir: Path) -> None:
    path = Path(f"{tmpdir}/model.pkl")
    network = MLP(input_size=2, hidden_layer_sizes=[3, 4], output_size=2)

    network.save(path)
    other = MLP(input_size=2, hidden_layer_sizes=[3, 4], output_size=2)
    other.load(path)

    assert network.layers[0].weights.tolist() == other.layers[0].weights.tolist()
    assert network.layers[1].weights.tolist() == other.layers[1].weights.tolist()
    assert network.output_layer.weights.tolist() == other.output_layer.weights.tolist()
    assert network.layers[0].biases.tolist() == other.layers[0].biases.tolist()
    assert network.layers[1].biases.tolist() == other.layers[1].biases.tolist()
