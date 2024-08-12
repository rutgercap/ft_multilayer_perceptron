from pathlib import Path

import numpy as np

from dataset import dataframe_to_numpy, dataset_from_path, normalize_data
from network import MLP


def one_hot_encode(y):
    array = np.array([[1, 0] if i == "B" else [0, 1] for i in y])
    return array


if __name__ == "__main__":
    print("Splitting data set...")
    df = dataset_from_path(Path("train.csv"))
    df = normalize_data(df)
    X, y = dataframe_to_numpy(df, target="diagnosis", index=True)
    print(len(y), len(X))
    print(X[1])
    model = MLP(input_size=X.shape[1], hidden_layer_sizes=[24, 24, 24], output_size=2)
    y = one_hot_encode(y)
    model.train(X, y, learning_rate=0.01, epochs=1000)
    print("Training complete.")
    model.save("model.pkl")
    print("Saved weights to model.pkl")
