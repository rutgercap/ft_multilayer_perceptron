from pathlib import Path
import pandas as pd
from pandas import read_csv
from numpy import ndarray, number, array
from sklearn.preprocessing import RobustScaler  # type: ignore

def one_hot_encode(y):
    return array([[1, 0] if i == "B" else [0, 1] for i in y])


def prep_data(path: Path) -> tuple[ndarray, ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = read_csv(path)
    df = normalize_data(df)
    X, y = dataframe_to_numpy(df, target="diagnosis", index=True)
    y = one_hot_encode(y)
    return X, y


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    column_numbers = df.select_dtypes(include=[number]).columns
    scaler = RobustScaler()
    df[column_numbers] = scaler.fit_transform(df[column_numbers])
    return df


def dataframe_to_numpy(
    df: pd.DataFrame, target: str, index: bool = False
) -> tuple[ndarray, ndarray]:
    if index:
        df.drop(columns=["index"], inplace=True)
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()
    return X, y
