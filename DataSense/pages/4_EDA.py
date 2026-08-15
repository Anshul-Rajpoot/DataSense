import streamlit as st
import matplotlib.pyplot as plt

from modules.eda_engine import categorical_distribution, correlation_matrix, dataset_statistics, generate_insights, numeric_summary, strong_correlations
from utils.helpers import get_working_dataframe


df = get_working_dataframe()
st.title("EDA")

if df is None:
    st.info("Upload a dataset first.")
    st.stop()

stats = dataset_statistics(df)
st.subheader("Dataset Statistics")
if stats:
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{stats.get('rows', 0):,}")
    b.metric("Columns", f"{stats.get('columns', 0):,}")
    c.metric("Numerical", f"{stats.get('numeric_columns', 0):,}")
    d.metric("Categorical", f"{stats.get('categorical_columns', 0):,}")

numeric_summary_df = numeric_summary(df)
if not numeric_summary_df.empty:
    selected_numeric = st.selectbox("Select numeric column", numeric_summary_df["column"].tolist())
    numeric_series = df[selected_numeric].dropna()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(numeric_series, bins=20, color="#2563eb", alpha=0.85)
        ax.set_title(f"Histogram: {selected_numeric}")
        ax.set_xlabel(selected_numeric)
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True)
    with chart_col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.boxplot(numeric_series, vert=True)
        ax.set_title(f"Box Plot: {selected_numeric}")
        ax.set_ylabel(selected_numeric)
        st.pyplot(fig, clear_figure=True)

    st.dataframe(numeric_summary_df, use_container_width=True)
else:
    st.info("No numeric columns available for histogram or box plot analysis.")

categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
if categorical_columns:
    st.subheader("Categorical Analysis")
    selected_category = st.selectbox("Select categorical column", categorical_columns)
    category_distribution = categorical_distribution(df, selected_category)
    st.dataframe(category_distribution, use_container_width=True)
    st.bar_chart(category_distribution.set_index("category")["count"], use_container_width=True)
else:
    st.info("No categorical columns available for categorical analysis.")

corr = correlation_matrix(df)
if corr is not None:
    st.subheader("Correlation Matrix")
    st.dataframe(corr, use_container_width=True)
    strong = strong_correlations(df)
    if not strong.empty:
        st.subheader("Strong Correlations")
        st.dataframe(strong, use_container_width=True)
else:
    st.info("No numeric columns available for correlation analysis.")

st.subheader("Automated Insights")
insights = generate_insights(df)
if insights:
    for insight in insights:
        st.write(f"• {insight}")
else:
    st.info("No rule-based insights were generated for this dataset.")
