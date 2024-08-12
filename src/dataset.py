from pathlib import Path

import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import RobustScaler  # type: ignore


def dataset_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, header=None)
    columns = list(df.columns)
    columns[0] = "index"
    columns[1] = "diagnosis"
    for i in range(2, len(columns)):
        columns[i] = f"feature_{i - 2}"
    df.columns = columns  # type: ignore
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def dataframe_to_numpy(
    df: pd.DataFrame, target: str, index: bool = False
) -> tuple[ndarray, ndarray]:
    if index:
        df = df.reset_index(drop=True)
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()
    return X, y
