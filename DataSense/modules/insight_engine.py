from __future__ import annotations

import pandas as pd

from modules.duplicates import duplicate_summary
from modules.eda_engine import generate_insights, strong_correlations
from modules.missing_values import missing_value_summary
from modules.outliers import outlier_summary
from modules.profiler import profile_dataframe


def build_report(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"status": "empty"}

    missing = missing_value_summary(df)
    return {
        "profile": profile_dataframe(df),
        "duplicates": duplicate_summary(df),
        "missing_columns": missing.loc[missing["missing_count"] > 0, "column"].tolist(),
        "strong_correlations": strong_correlations(df).to_dict(orient="records"),
        "outliers": outlier_summary(df).to_dict(orient="records"),
        "insights": generate_insights(df),
    }
