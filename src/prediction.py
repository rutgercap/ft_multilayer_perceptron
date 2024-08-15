from pathlib import Path
from sys import argv
from numpy import ndarray, mean, log

from network import MLP
from dataset import prep_data

def binary_cross_entropy(y_true: ndarray, y_pred: ndarray):
    return -mean(
        y_true * log(y_pred + 1e-15) + (1 - y_true) * log(1 - y_pred + 1e-15)
    )

def accuracy(y_true: ndarray, y_pred: ndarray):
    return mean(y_true == y_pred)

def main():
    if len(argv) != 3:
        print("Usage: python prediction <model.json> <data.csv>")
        exit(1)
    model_path = argv[1]
    data_path = argv[2]
    model = MLP.from_file(Path(model_path))
    X, y = prep_data(Path(data_path))
    prediction = model.predict(X)
    loss = binary_cross_entropy(y, prediction)
    found_accuracy = accuracy(y, prediction)
    print(f"Accuracy: {found_accuracy} | loss: {loss}")



if __name__ == "__main__":
    main()
