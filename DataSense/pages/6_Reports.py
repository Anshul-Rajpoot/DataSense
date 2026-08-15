import streamlit as st
from io import BytesIO
import json
import zipfile

import pandas as pd

from modules.insight_engine import build_report
from utils.helpers import dataframe_download_csv, get_working_dataframe, store_report


df = get_working_dataframe()
st.title("Reports")

if df is None:
    st.info("Upload a dataset first.")
    st.stop()

report = build_report(df)
store_report(report)

st.subheader("Reports & Downloads")

csv_bytes = dataframe_download_csv(df)
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="CleanedData")
excel_buffer.seek(0)

report_json = json.dumps(report, indent=2, default=str)
model_report = st.session_state.get("last_model_report")

def _make_download_all_bundle() -> bytes:
    bundle = BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("datasense_cleaned.csv", csv_bytes)
        archive.writestr("datasense_cleaned.xlsx", excel_buffer.getvalue())
        archive.writestr("datasense_eda_report.json", report_json)
        if model_report:
            archive.writestr("datasense_model_results.json", json.dumps(model_report, indent=2, default=str))
    bundle.seek(0)
    return bundle.getvalue()

left, right = st.columns(2)
left.download_button("Download Cleaned CSV", data=csv_bytes, file_name="datasense_cleaned.csv", mime="text/csv")
right.download_button(
    "Download Cleaned Excel",
    data=excel_buffer.getvalue(),
    file_name="datasense_cleaned.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.download_button("Download EDA Report", data=report_json, file_name="datasense_eda_report.json", mime="application/json")

if model_report:
    st.download_button(
        "Download Model Results",
        data=json.dumps(model_report, indent=2, default=str),
        file_name="datasense_model_results.json",
        mime="application/json",
    )

st.download_button(
    "Download All",
    data=_make_download_all_bundle(),
    file_name="datasense_export_bundle.zip",
    mime="application/zip",
)

st.json(report)
