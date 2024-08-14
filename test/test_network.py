from pathlib import Path

import numpy
from numpy import array
from pytest import raises

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
    numpy.random.seed(0)
    output_size = 2
    network = MLP(input_size=2, hidden_layer_sizes=[3, 4], output_size=output_size)
    X = array([[1, 2], [3, 4], [5, 6]])
    y = array([[1, 0], [0, 1], [1, 0]])
    learning_rate = 0.1
    epochs = 10

    result = network.forward([1, 2])
    network.train((X, y), learning_rate, epochs, silent=True)

    assert len(result[0]) == output_size
    assert numpy.allclose(
        network.layers[0].weights,
        [[1.76406309, 0.40028914, 0.98080656], [2.24091469, 1.8678221, -0.97267574]],
    )
    assert numpy.allclose(
        network.layers[0].biases, [[1.07477400e-05, 1.32180141e-04, 2.53356336e-03]]
    )
    assert numpy.allclose(
        network.layers[1].weights,
        [
            [0.95200566, -0.16091427, -0.04345054, 0.4922834],
            [0.14531307, 1.44792475, 0.80124633, 0.1760041],
            [0.44386323, 0.33367433, 1.49407907, -0.20515826],
        ],
    )
    assert numpy.allclose(
        network.layers[1].biases, [[0.00030703, -0.00156479, 0.01062937, 0.01348533]]
    )
    assert numpy.allclose(
        network.output_layer.weights,
        [
            [-0.19132369, -0.34970435],
            [-2.89479723, 0.99542601],
            [0.59534248, -0.47307131],
            [1.9199237, -1.10453475],
        ],
    )
    assert numpy.allclose(network.output_layer.biases, [[-0.03446333, 0.03446333]])


def test_can_save_weights_to_file(tmpdir: Path) -> None:
    path = Path(f"{tmpdir}/model.json")
    network = MLP(input_size=2, hidden_layer_sizes=[2, 3], output_size=3)

    network.save(path)
    other = MLP.from_file(path)

    assert network.input_size == other.input_size
    assert network.output_layer.weights.tolist() == other.output_layer.weights.tolist()
    assert network.layers[0].weights.tolist() == other.layers[0].weights.tolist()
    assert network.layers[0].biases.tolist() == other.layers[0].biases.tolist()
    assert network.layers[1].weights.tolist() == other.layers[1].weights.tolist()
    assert network.layers[1].biases.tolist() == other.layers[1].biases.tolist()


def test_predict_and_forward_give_similar_results() -> None:
    network = MLP(input_size=2, hidden_layer_sizes=[3, 4], output_size=2)
    X = array([[1, 2], [3, 4], [5, 6]])

    forward_result = network.forward(X)
    predict_result = network.predict(X)

    assert numpy.allclose(forward_result, predict_result)
