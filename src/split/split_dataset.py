from pathlib import Path

import pandas as pd


def split(path: Path) -> None:
    print("Splitting data set...")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv("your_file.csv")


if __name__ == "__main__":
    split(Path("data.csv"))
