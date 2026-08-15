from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_data(source: Any) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()

    name = getattr(source, "name", "")
    suffix = Path(name).suffix.lower()

    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            pass

    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(source, engine="openpyxl")
        return pd.read_csv(source)
    except Exception:
        return pd.DataFrame()
