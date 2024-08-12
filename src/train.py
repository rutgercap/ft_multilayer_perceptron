from pathlib import Path

import numpy as np

from dataset import dataframe_to_numpy, dataset_from_path, normalize_data
from network import MLP


def one_hot_encode(y):
    array = np.array([[1, 0] if i == "B" else [0, 1] for i in y])
    return array

def prep_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = dataset_from_path(path)
    df = normalize_data(df)
    X, y = dataframe_to_numpy(df, target="diagnosis", index=True)
    y = one_hot_encode(y)
    return X, y

if __name__ == "__main__":
    print("Splitting data set...")
    X, y = prep_data(Path("train.csv"))
    X_val, y_val = prep_data(Path("test.csv"))
    model = MLP(input_size=X.shape[1], hidden_layer_sizes=[24, 24, 24], output_size=2)
    history = model.train((X, y), learning_rate=0.001, epochs=1000, validation_data=(X_val, y_val))
    print("Training complete.")
    model.save("model.pkl")
    print("Saved weights to model.pkl")
