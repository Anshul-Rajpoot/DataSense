from __future__ import annotations

import pandas as pd


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna().astype(str)

    if non_null.empty:
        return "unknown"

    # Try numeric detection
    numeric_candidates = pd.to_numeric(
        non_null,
        errors="coerce"
    )

    if numeric_candidates.notna().mean() >= 0.8:
        return "numeric"

    # Try datetime detection
    parsed_datetime = pd.to_datetime(
        non_null,
        errors="coerce"
    )

    if parsed_datetime.notna().mean() >= 0.8:
        return "datetime"

    return "categorical"


def detect_invalid_examples(series: pd.Series, target_type: str, limit: int = 3) -> list[str]:
    if target_type == "numeric":
        invalid = pd.to_numeric(series, errors="coerce").isna() & series.notna()
    elif target_type == "datetime":
        invalid = pd.to_datetime(series, errors="coerce").isna() & series.notna()
    else:
        return []

    return series[invalid].astype(str).head(limit).tolist()


def datatype_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "current_dtype", "detected_dtype", "non_null_count", "invalid_examples", "recommendation"])

    rows = []
    for column in df.columns:
        series = df[column]
        detected = _infer_column_type(series)
        invalid_examples = detect_invalid_examples(series, detected)
        recommendation = None
        if str(series.dtype) == "object" and detected in {"numeric", "datetime"}:
            recommendation = f"Convert to {detected}"
        rows.append(
            {
                "column": column,
                "current_dtype": str(series.dtype),
                "detected_dtype": detected,
                "non_null_count": int(series.notna().sum()),
                "invalid_examples": ", ".join(invalid_examples) if invalid_examples else "—",
                "recommendation": recommendation or "—",
            }
        )

    return pd.DataFrame(rows)


def convert_column(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()

    converted = df.copy()
    if target_type == "numeric":
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    elif target_type == "datetime":
        converted[column] = pd.to_datetime(converted[column], errors="coerce")
    else:
        converted[column] = converted[column].astype("string")
    return converted
