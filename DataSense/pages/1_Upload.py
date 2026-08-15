import streamlit as st

from modules.loader import load_data
from utils.constants import SUPPORTED_FILE_TYPES
from utils.helpers import dataframe_metrics, ensure_session_state, set_dataset


ensure_session_state()

st.title("Upload")
st.write("Load a dataset to begin the analysis workflow.")
st.markdown("""
<div style="padding:1rem 1.25rem;border:1px solid rgba(148,163,184,0.35);border-radius:1rem;">
<strong>Drag and drop your file here.</strong><br/>
Supported formats: XLSX, XLS, CSV.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_FILE_TYPES)
if uploaded_file is not None:
    dataframe = load_data(uploaded_file)
    if dataframe.empty:
        st.warning("No rows were loaded from the selected file.")
    else:
        set_dataset(dataframe, uploaded_file.name)
        metrics = dataframe_metrics(dataframe)
        st.success(f"Loaded {len(dataframe):,} rows and {len(dataframe.columns):,} columns.")
        left, right, middle = st.columns(3)
        left.metric("Rows", f"{metrics['rows']:,}")
        middle.metric("Columns", f"{metrics['columns']:,}")
        right.metric("Missing cells", f"{metrics['missing_cells']:,}")
        st.dataframe(dataframe.head(20), use_container_width=True)
