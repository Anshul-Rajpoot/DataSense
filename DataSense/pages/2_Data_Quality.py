import streamlit as st

from modules.duplicates import duplicate_summary
from modules.missing_values import missing_value_summary
from modules.datatype import datatype_summary
from modules.preprocessing import clean_dataframe
from utils.helpers import dataframe_metrics, get_original_dataframe, get_working_dataframe, store_dataframe


df = get_working_dataframe()
st.title("Data Quality")

if df is None:
    st.info("Upload a dataset first.")
    st.stop()

metrics = dataframe_metrics(df)
original_df = get_original_dataframe()

st.subheader("Dataset Overview")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Rows", f"{metrics['rows']:,}")
col2.metric("Columns", f"{metrics['columns']:,}")
col3.metric("Duplicates", f"{metrics['duplicate_rows']:,}")
col4.metric("Missing", f"{metrics['missing_cells']:,}")
col5.metric("Numerical", f"{metrics['numeric_columns']:,}")
col6.metric("Categorical", f"{metrics['categorical_columns']:,}")

st.subheader("Column Analysis")
missing = missing_value_summary(df)
types = datatype_summary(df)
analysis = types.merge(missing[["column", "missing_count", "missing_percent", "recommended_strategy"]], on="column", how="left")
analysis["unique_values"] = [int(df[column].nunique(dropna=True)) for column in analysis["column"]]
st.dataframe(
    analysis[["column", "current_dtype", "detected_dtype", "missing_count", "missing_percent", "unique_values", "recommendation", "recommended_strategy"]],
    use_container_width=True,
)

duplicate_info = duplicate_summary(df)
if duplicate_info["duplicate_rows"] > 0:
    percentage = round(duplicate_info["duplicate_rows"] / len(df) * 100, 2)
    st.warning(
        f"{duplicate_info['duplicate_rows']} duplicate rows detected ({percentage}% of the dataset)."
    )
    duplicate_rows = df[df.duplicated(keep=False)]
    with st.expander("View duplicate records"):
        st.dataframe(duplicate_rows, use_container_width=True)

    remove_col, keep_col = st.columns(2)
    if remove_col.button("Remove Duplicates", use_container_width=True):
        cleaned = clean_dataframe(df, drop_duplicates=True)
        store_dataframe(cleaned)
        st.success("Duplicate rows removed from the working dataset.")
        st.rerun()
    if keep_col.button("Keep Them", use_container_width=True):
        st.info("The working dataset was left unchanged.")
else:
    st.success("No duplicate rows were detected.")

if original_df is not None:
    st.caption(f"Original rows: {len(original_df):,} · Working rows: {len(df):,}")

st.subheader("Missing Values")
st.dataframe(missing, use_container_width=True)
