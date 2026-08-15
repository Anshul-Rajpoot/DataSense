from __future__ import annotations

import pandas as pd


MISSING_PERCENT_THRESHOLD = 5.0
SKEW_THRESHOLD = 1.0


def _outlier_percentage(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((numeric < lower) | (numeric > upper)).mean() * 100)


def recommend_imputation(series: pd.Series) -> dict:
    non_null = series.dropna()
    if non_null.empty:
        return {
            "strategy": "leave unchanged",
            "reason": "No non-null values are available for inference.",
        }

    if pd.api.types.is_numeric_dtype(series):
        skewness = float(non_null.skew()) if len(non_null) > 2 else 0.0
        outlier_percentage = _outlier_percentage(non_null)
        if outlier_percentage > 5 or abs(skewness) > SKEW_THRESHOLD:
            return {
                "strategy": "median",
                "reason": "Distribution is skewed or contains notable outliers.",
                "skewness": round(skewness, 3),
                "outlier_percentage": round(outlier_percentage, 2),
            }
        return {
            "strategy": "mean",
            "reason": "Distribution is relatively stable and near symmetric.",
            "skewness": round(skewness, 3),
            "outlier_percentage": round(outlier_percentage, 2),
        }

    mode = non_null.mode(dropna=True)
    return {
        "strategy": "mode",
        "reason": "Categorical columns are usually best filled with the most frequent value.",
        "top_value": mode.iloc[0] if not mode.empty else None,
    }


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "dtype", "missing_count", "missing_percent", "recommended_strategy", "reason"])

    rows = []
    for column in df.columns:
        series = df[column]
        recommendation = recommend_imputation(series)
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
                "recommended_strategy": recommendation["strategy"],
                "reason": recommendation["reason"],
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def missing_value_details(df: pd.DataFrame, column: str) -> dict:
    if df is None or df.empty or column not in df.columns:
        return {}

    series = df[column]
    recommendation = recommend_imputation(series)
    return {
        "column": column,
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_percent": round(float(series.isna().mean() * 100), 2),
        **recommendation,
    }
