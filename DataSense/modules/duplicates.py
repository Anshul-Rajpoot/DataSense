from __future__ import annotations

import pandas as pd


def duplicate_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"duplicate_rows": 0, "duplicate_columns": 0}

    duplicate_rows = int(df.duplicated().sum())
    duplicate_columns = int(df.T.duplicated().sum())
    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_columns": duplicate_columns,
    }
