from __future__ import annotations

import pandas as pd

from modules.outliers import detect_outliers, iqr_bounds


def fill_missing_values(df: pd.DataFrame, strategy: str = "none") -> pd.DataFrame:
    if df is None or df.empty or strategy == "none":
        return df.copy() if df is not None else pd.DataFrame()

    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    categorical_columns = cleaned.select_dtypes(exclude="number").columns

    if strategy == "drop":
        return cleaned.dropna()

    if strategy == "mean":
        for column in numeric_columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
    elif strategy == "median":
        for column in numeric_columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    elif strategy == "mode":
        for column in cleaned.columns:
            mode = cleaned[column].mode(dropna=True)
            if not mode.empty:
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])

    for column in categorical_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].mode(dropna=True).iloc[0] if not cleaned[column].mode(dropna=True).empty else "Unknown")

    return cleaned


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cleaned = df.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    return cleaned


def apply_missing_strategy(df: pd.DataFrame, column: str, strategy: str, constant_value: str | None = None) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()

    cleaned = df.copy()
    if strategy == "delete rows":
        return cleaned.dropna(subset=[column])
    if strategy == "mean":
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(pd.to_numeric(cleaned[column], errors="coerce").mean())
    elif strategy == "median":
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(pd.to_numeric(cleaned[column], errors="coerce").median())
    elif strategy == "mode":
        mode = cleaned[column].mode(dropna=True)
        if not mode.empty:
            cleaned[column] = cleaned[column].fillna(mode.iloc[0])
    elif strategy == "forward fill":
        cleaned[column] = cleaned[column].fillna(method="ffill")
    elif strategy == "backward fill":
        cleaned[column] = cleaned[column].fillna(method="bfill")
    elif strategy == "constant value" and constant_value is not None:
        cleaned[column] = cleaned[column].fillna(constant_value)
    return cleaned


def convert_column_dtype(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()

    cleaned = df.copy()
    if target_type == "numeric":
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    elif target_type == "datetime":
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
    return cleaned


def handle_outliers(df: pd.DataFrame, column: str, action: str, method: str = "iqr") -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()

    cleaned = df.copy()
    outlier_info = detect_outliers(cleaned, column, method=method)
    mask = outlier_info["mask"]

    if action == "remove outliers":
        return cleaned.loc[~mask].copy()
    if action == "cap outliers":
        lower, upper = iqr_bounds(cleaned[column])
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").clip(lower=lower, upper=upper)
    return cleaned
