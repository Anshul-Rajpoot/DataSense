from __future__ import annotations

import pandas as pd

from modules.missing_values import missing_value_summary
from modules.outliers import outlier_summary


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().transpose().reset_index(names="column")


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    categorical = df.select_dtypes(exclude="number")
    if categorical.empty:
        return pd.DataFrame()
    rows = []
    for column in categorical.columns:
        series = categorical[column]
        rows.append(
            {
                "column": column,
                "unique_values": int(series.nunique(dropna=True)),
                "top_value": series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else None,
                "missing_count": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    return numeric.corr(numeric_only=True)


def dataset_statistics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}

    numeric = df.select_dtypes(include="number")
    stats = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(numeric.shape[1]),
        "categorical_columns": int(df.select_dtypes(exclude="number").shape[1]),
    }
    if not numeric.empty:
        stats.update(
            {
                "mean_row": numeric.mean(numeric_only=True).round(2).to_dict(),
                "median_row": numeric.median(numeric_only=True).round(2).to_dict(),
                "std_row": numeric.std(numeric_only=True).round(2).to_dict(),
                "min_row": numeric.min(numeric_only=True).round(2).to_dict(),
                "max_row": numeric.max(numeric_only=True).round(2).to_dict(),
            }
        )
    return stats


def strong_correlations(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    corr = correlation_matrix(df)
    if corr is None:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    rows = []
    columns = corr.columns.tolist()
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_1": left, "feature_2": right, "correlation": round(float(value), 3)})
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    return pd.DataFrame(rows).sort_values("correlation", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)


def generate_insights(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    insights: list[str] = []
    missing = missing_value_summary(df)
    missing_columns = missing.loc[missing["missing_count"] > 0]
    if not missing_columns.empty:
        high_missing = int((missing_columns["missing_percent"] > 5).sum())
        if high_missing:
            insights.append(f"{high_missing} columns contain missing values above 5%.")

    corr = strong_correlations(df)
    for _, row in corr.head(3).iterrows():
        insights.append(f"{row['feature_1']} and {row['feature_2']} have a correlation of {row['correlation']}.")

    if not df.select_dtypes(include="number").empty:
        skewed = df.select_dtypes(include="number").skew(numeric_only=True).abs()
        skewed_columns = skewed[skewed > 1].index.tolist()
        if skewed_columns:
            insights.append(f"{', '.join(skewed_columns[:3])} appear to be skewed.")

    categorical = df.select_dtypes(exclude="number")
    for column in categorical.columns[:3]:
        unique_count = int(categorical[column].nunique(dropna=True))
        if unique_count > 0:
            insights.append(f"{column} contains {unique_count} unique categories.")

    outliers = outlier_summary(df)
    if not outliers.empty:
        top_outlier = outliers.iloc[0]
        if int(top_outlier["outlier_count"]) > 0:
            insights.append(f"{top_outlier['column']} contains {int(top_outlier['outlier_count'])} potential outliers.")

    return insights


def categorical_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame(columns=["category", "count", "percentage"])

    series = df[column].astype("string").fillna("Missing")
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = ["category", "count"]
    counts["percentage"] = (counts["count"] / len(series) * 100).round(2)
    return counts
