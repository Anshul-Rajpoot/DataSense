from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from utils.constants import (
    HISTORY_KEY,
    LAST_REPORT_KEY,
    ORIGINAL_DF_KEY,
    UPLOADED_FILENAME_KEY,
    WORKING_DF_KEY,
)


def set_page_config() -> None:
    st.set_page_config(page_title="DataSense", page_icon="DS", layout="wide")


def ensure_session_state() -> None:
    st.session_state.setdefault(ORIGINAL_DF_KEY, None)
    st.session_state.setdefault(WORKING_DF_KEY, None)
    st.session_state.setdefault(UPLOADED_FILENAME_KEY, None)
    st.session_state.setdefault(HISTORY_KEY, [])
    st.session_state.setdefault(LAST_REPORT_KEY, None)


def set_dataset(df: pd.DataFrame, filename: str | None = None) -> None:
    ensure_session_state()
    original_df = df.copy(deep=True)
    working_df = df.copy(deep=True)
    st.session_state[ORIGINAL_DF_KEY] = original_df
    st.session_state[WORKING_DF_KEY] = working_df
    st.session_state[UPLOADED_FILENAME_KEY] = filename
    st.session_state[HISTORY_KEY] = [working_df.copy(deep=True)]


def store_dataframe(df: pd.DataFrame) -> None:
    ensure_session_state()
    st.session_state[WORKING_DF_KEY] = df.copy(deep=True)
    st.session_state[HISTORY_KEY].append(df.copy(deep=True))


def get_original_dataframe() -> pd.DataFrame | None:
    ensure_session_state()
    return st.session_state.get(ORIGINAL_DF_KEY)


def get_working_dataframe() -> pd.DataFrame | None:
    ensure_session_state()
    return st.session_state.get(WORKING_DF_KEY)


def get_current_dataframe() -> pd.DataFrame | None:
    return get_working_dataframe()


def undo_last_change() -> pd.DataFrame | None:
    ensure_session_state()
    history = st.session_state[HISTORY_KEY]
    if len(history) <= 1:
        return st.session_state.get(WORKING_DF_KEY)

    history.pop()
    restored = history[-1].copy(deep=True)
    st.session_state[WORKING_DF_KEY] = restored
    return restored


def reset_to_original() -> pd.DataFrame | None:
    ensure_session_state()
    original_df = st.session_state.get(ORIGINAL_DF_KEY)
    if original_df is None:
        return None
    restored = original_df.copy(deep=True)
    st.session_state[WORKING_DF_KEY] = restored
    st.session_state[HISTORY_KEY] = [restored.copy(deep=True)]
    return restored


def dataframe_metrics(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "missing_cells": 0,
            "duplicate_rows": 0,
            "numeric_columns": 0,
            "categorical_columns": 0,
        }

    numeric_columns = df.select_dtypes(include="number").shape[1]
    categorical_columns = df.select_dtypes(exclude="number").shape[1]
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(numeric_columns),
        "categorical_columns": int(categorical_columns),
    }


def dataframe_preview_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    metrics = dataframe_metrics(df)
    return (
        f"<p>{metrics['rows']:,} rows and {metrics['columns']:,} columns loaded.</p>"
        f"<p>{metrics['missing_cells']:,} missing cells · {metrics['duplicate_rows']:,} duplicate rows</p>"
    )


def dataframe_download_csv(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def store_report(report: dict | None) -> None:
    ensure_session_state()
    st.session_state[LAST_REPORT_KEY] = report


def get_last_report() -> dict | None:
    ensure_session_state()
    return st.session_state.get(LAST_REPORT_KEY)
