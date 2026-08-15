from __future__ import annotations

import pandas as pd


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return (float("nan"), float("nan"))
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def zscore_outlier_count(series: pd.Series, threshold: float = 3.0) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0
    std = numeric.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0
    zscores = (numeric - numeric.mean()) / std
    return int((zscores.abs() > threshold).sum())


def detect_outliers(df: pd.DataFrame, column: str, method: str = "iqr") -> dict:
    if df is None or df.empty or column not in df.columns:
        return {"count": 0, "percentage": 0.0, "mask": pd.Series(dtype=bool)}

    series = pd.to_numeric(df[column], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return {"count": 0, "percentage": 0.0, "mask": pd.Series(False, index=df.index)}

    if method == "z-score":
        std = valid.std(ddof=0)
        if std == 0 or pd.isna(std):
            mask = pd.Series(False, index=df.index)
        else:
            zscores = (series - valid.mean()) / std
            mask = zscores.abs() > 3.0
    else:
        lower, upper = iqr_bounds(series)
        mask = (series < lower) | (series > upper)

    count = int(mask.fillna(False).sum())
    percentage = round(float(count / len(df) * 100), 2) if len(df) else 0.0
    return {"count": count, "percentage": percentage, "mask": mask.fillna(False)}


def outlier_summary(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "method", "outlier_count", "outlier_percent", "lower_bound", "upper_bound"])

    rows = []
    for column in df.select_dtypes(include="number").columns:
        if method == "z-score":
            count = zscore_outlier_count(df[column])
            lower, upper = float("nan"), float("nan")
        else:
            lower, upper = iqr_bounds(df[column])
            count = int(((pd.to_numeric(df[column], errors="coerce") < lower) | (pd.to_numeric(df[column], errors="coerce") > upper)).sum())
        rows.append(
            {
                "column": column,
                "method": method.upper(),
                "outlier_count": count,
                "outlier_percent": round(float(count / len(df) * 100), 2) if len(df) else 0.0,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
    return pd.DataFrame(rows).sort_values(["outlier_count", "column"], ascending=[False, True]).reset_index(drop=True)
