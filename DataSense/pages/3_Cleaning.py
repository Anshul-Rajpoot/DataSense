import streamlit as st

from modules.datatype import datatype_summary
from modules.missing_values import missing_value_details, missing_value_summary
from modules.outliers import detect_outliers, outlier_summary
from modules.preprocessing import apply_missing_strategy, convert_column_dtype, handle_outliers
from utils.helpers import get_working_dataframe, reset_to_original, store_dataframe, undo_last_change


df = get_working_dataframe()
st.title("Cleaning")

if df is None:
    st.info("Upload a dataset first.")
    st.stop()

left, right = st.columns(2)
if left.button("Undo last change"):
    restored = undo_last_change()
    if restored is not None:
        st.success("Reverted to the previous working dataset state.")
        st.rerun()
if right.button("Reset to original"):
    restored = reset_to_original()
    if restored is not None:
        st.success("Working dataset reset to the original upload.")
        st.rerun()

missing_summary = missing_value_summary(df)
st.subheader("Missing Value Analysis")
st.dataframe(missing_summary, use_container_width=True)

missing_columns = missing_summary.loc[missing_summary["missing_count"] > 0, "column"].tolist()
if missing_columns:
    selected_missing_column = st.selectbox("Select a column to clean", missing_columns)
    details = missing_value_details(df, selected_missing_column)
    st.write(
        f"Missing values: {details.get('missing_count', 0)} ({details.get('missing_percent', 0)}%). Recommended strategy: {details.get('strategy', 'leave unchanged')}"
    )
    st.caption(details.get("reason", ""))

    strategy = st.radio(
        "Choose action",
        [
            "delete rows",
            "mean",
            "median",
            "mode",
            "forward fill",
            "backward fill",
            "constant value",
            "leave unchanged",
        ],
        horizontal=True,
    )
    constant_value = None
    if strategy == "constant value":
        constant_value = st.text_input("Constant value")
    if st.button("Apply missing value action"):
        cleaned = apply_missing_strategy(df, selected_missing_column, strategy, constant_value=constant_value)
        store_dataframe(cleaned)
        st.success("Missing value strategy applied to the working dataset.")
        st.rerun()
else:
    st.success("No missing values were detected.")

numeric_columns = df.select_dtypes(include="number").columns.tolist()
st.subheader("Outlier Detection")
if numeric_columns:
    selected_outlier_column = st.selectbox("Select numeric column", numeric_columns)
    outlier_method = st.radio("Method", ["iqr", "z-score"], horizontal=True)
    outlier_info = detect_outliers(df, selected_outlier_column, method=outlier_method)
    st.write(
        f"Outliers found: {outlier_info['count']} ({outlier_info['percentage']}%)"
    )
    action = st.radio("What would you like to do?", ["keep outliers", "remove outliers", "cap outliers"], horizontal=True)
    if st.button("Apply outlier action"):
        cleaned = handle_outliers(df, selected_outlier_column, action, method=outlier_method)
        store_dataframe(cleaned)
        st.success("Outlier action applied to the working dataset.")
        st.rerun()
    st.dataframe(outlier_summary(df, method=outlier_method), use_container_width=True)
else:
    st.info("No numeric columns were found for outlier analysis.")

st.subheader("Data Type Detection")
type_summary = datatype_summary(df)
st.dataframe(type_summary, use_container_width=True)

convertible = type_summary.loc[type_summary["recommendation"] != "—", "column"].tolist()
if convertible:
    selected_type_column = st.selectbox("Select a column to convert", convertible)
    selected_row = type_summary[type_summary["column"] == selected_type_column].iloc[0]
    target_type = selected_row["detected_dtype"]
    st.write(f"Current datatype: {selected_row['current_dtype']} · Detected: {target_type}")
    st.caption(f"Invalid values: {selected_row['invalid_examples']}")
    if st.button(f"Convert to {target_type}"):
        cleaned = convert_column_dtype(df, selected_type_column, target_type)
        store_dataframe(cleaned)
        st.success("Column converted in the working dataset.")
        st.rerun()
