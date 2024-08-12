import pandas as pd
from numpy import ndarray, number
from sklearn.preprocessing import RobustScaler  # type: ignore


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
