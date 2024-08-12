import random
from pathlib import Path

from pandas import DataFrame, read_csv


def dataset_from_path(path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = read_csv(path, header=None)
    columns = list(df.columns)
    columns[0] = "index"
    columns[1] = "diagnosis"
    for i in range(2, len(columns)):
        columns[i] = f"feature_{i - 2}"
    df.columns = columns  # type: ignore
    return df


def split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    test_size = int(len(df) * 0.2)
    others = [i for i in range(len(df))]
    indices = random.sample(range(len(df)), test_size)
    for i in indices:
        others.remove(i)
    test = df.iloc[indices]
    train = df.iloc[others]
    return test, train


if __name__ == "__main__":
    print("Splitting data set...")
    df = dataset_from_path(Path("data.csv"))
    test, train = split(df)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    print("Data set split successfully.")
