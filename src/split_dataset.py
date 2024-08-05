from pathlib import Path
from dataset import dataset_from_path


def split(path: Path) -> None:
    df = dataset_from_path(path)  
    test_size = int(len(df) * 0.2)
    test, train = df.iloc[:test_size], df.iloc[test_size:]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    print("Splitting data set...")
    split(Path("data.csv"))
    print("Data set split successfully.")
