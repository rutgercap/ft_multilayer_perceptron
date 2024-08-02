from pathlib import Path
from sys import argv

import pandas as pd


def split(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, header=None)
    columns = list(df.columns)
    columns[0] = "index"
    columns[1] = "diagnosis"
    df.columns = columns
    test_size = int(len(df) * 0.2)
    test, train = df.iloc[:test_size], df.iloc[test_size:]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    if len(argv) < 2:
        raise ValueError("Please provide the path to the data set.")
    # split(Path(argv[1]))
    print("Splitting data set...")
    split(Path("data.csv"))
    print("Data set split successfully.")
