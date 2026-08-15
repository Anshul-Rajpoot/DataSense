import streamlit as st

from utils.constants import APP_SUBTITLE
from utils.helpers import dataframe_metrics, get_working_dataframe, set_page_config


st.sidebar.title("DataSense ")
st.sidebar.markdown("""
### 👨‍💻 Developer

**Anshul Rajpoot**  
📘 Scholar No: `2311401168`  

🎓 Electronics & Communication Engineering  
🏛️ MANIT Bhopal
""")


set_page_config()


def _render_home() -> None:
    st.markdown(
        """
        <style>
        .datasense-hero {
            padding: 2rem;
            border-radius: 1.25rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
            color: white;
        }
        .datasense-hero h1 { margin-bottom: 0.25rem; }
        .datasense-hero p { margin: 0; opacity: 0.9; }
        </style>
        <div class="datasense-hero">
          <h1>DataSense</h1>
          <p>Automated data cleaning, EDA, and machine learning for tabular datasets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption(APP_SUBTITLE)

    st.markdown(
        """
        Use the sidebar pages to upload data, inspect quality issues, clean it with
        recommendations, explore relationships, train models, and export reports.
        """
    )

    current_df = get_working_dataframe()
    if current_df is None:
        st.info("Upload a dataset from the Upload page to begin.")
        return

    metrics = dataframe_metrics(current_df)
    st.subheader("Current Session")
    left, middle, right = st.columns(3)
    left.metric("Rows", f"{metrics['rows']:,}")
    middle.metric("Columns", f"{metrics['columns']:,}")
    right.metric("Missing cells", f"{metrics['missing_cells']:,}")

    st.dataframe(current_df.head(20), use_container_width=True)


_render_home()
