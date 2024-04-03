import pandas as pd
from typing import List
import math


def add_feature(data: pd.DataFrame) -> pd.DataFrame:
    return data


def preprocess(
    data: pd.DataFrame, exclude: List[str] = ["funding_rate", "log"]
) -> pd.DataFrame:
    columns_to_drop = []
    for column in data.columns:
        if not any(substring in column for substring in exclude):
            new_column_name = f"log_{column}"
            data[new_column_name] = data[column].apply(
                lambda x: math.log(x) if x > 0 else float("nan")
            )
            columns_to_drop.append(column)
    data = data.drop(columns=columns_to_drop)
    return data


def lagging(data: pd.DataFrame, lagging_length: int = 10) -> pd.DataFrame:
    columns = data.columns
    for column in columns:
        for lag in range(1, lagging_length + 1):
            lagged_column_name = f"{column}_lag_{lag}"
            data[lagged_column_name] = data[column].shift(lag)
    return data


def set_target(
    data: pd.DataFrame, target_maker: List[str] = ["close"], timegap: int = 1
) -> tuple[pd.DataFrame, List[str]]:
    target = []
    for column in data.columns:
        for target_prefix in target_maker:
            if target_prefix in column:
                target_column_name = f"target_{column}"
                data[target_column_name] = data[column].shift(-timegap)
                target.append(target_column_name)

    data.dropna(inplace=True)
    return data, target
