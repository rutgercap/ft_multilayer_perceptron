from pathlib import Path

from pandas import DataFrame

from dataset import dataset_from_path


def split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    test_size = int(len(df) * 0.2)
    test, train = df.iloc[:test_size], df.iloc[test_size:]
    return test, train


if __name__ == "__main__":
    print("Splitting data set...")
    df = dataset_from_path(Path("data.csv"))
    test, train = split(df)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    print("Data set split successfully.")
