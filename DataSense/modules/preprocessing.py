from __future__ import annotations

from collections.abc import Iterable

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


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy of ``df`` without the requested columns."""
    if df is None:
        return pd.DataFrame()

    columns_to_drop = list(columns)
    if not columns_to_drop:
        return df.copy()

    missing_columns = [column for column in columns_to_drop if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {', '.join(map(str, missing_columns))}")

    return df.drop(columns=columns_to_drop).copy()


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


def convert_column_dtype(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    errors: str = "coerce",
) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()

    cleaned = df.copy()
    if target_type in {"numeric", "float"}:
        cleaned[column] = pd.to_numeric(cleaned[column], errors=errors)
        if target_type == "float":
            cleaned[column] = cleaned[column].astype(float)
    elif target_type == "integer":
        numeric = pd.to_numeric(cleaned[column], errors=errors)
        if errors == "raise" and (numeric.dropna() % 1 != 0).any():
            raise ValueError("The column contains decimal values that cannot be converted to integer.")
        cleaned[column] = numeric.astype("Int64")
    elif target_type == "datetime":
        cleaned[column] = pd.to_datetime(cleaned[column], errors=errors)
    elif target_type == "boolean":
        boolean_values = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        }
        normalized = cleaned[column].astype("string").str.strip().str.lower()
        invalid = normalized.notna() & ~normalized.isin(boolean_values)
        if errors == "raise" and invalid.any():
            raise ValueError("The column contains values that are not valid booleans (true/false, yes/no, or 1/0).")
        cleaned[column] = normalized.map(boolean_values).astype("boolean")
    elif target_type == "string":
        cleaned[column] = cleaned[column].astype("string")
    else:
        raise ValueError(f"Unsupported target datatype: {target_type}")
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
