import streamlit as st

from modules.ml_engine import FAST_CLASSIFICATION_MODELS, FAST_REGRESSION_MODELS, get_candidate_models, infer_problem_type, train_model
from utils.helpers import get_working_dataframe


df = get_working_dataframe()
st.title("Modeling")

if df is None:
    st.info("Upload a dataset first.")
    st.stop()

target_column = st.selectbox("Select target column", df.columns.tolist())
problem_type = infer_problem_type(df[target_column])
st.caption(f"Detected problem type: {problem_type}")

include_heavy_models = st.checkbox("Include slower models", value=False)
available_models = list(get_candidate_models(problem_type, include_heavy_models=include_heavy_models).keys())
default_models = FAST_CLASSIFICATION_MODELS if problem_type == "classification" else FAST_REGRESSION_MODELS
selected_models = st.multiselect(
    "Model selection",
    available_models,
    default=[model for model in default_models if model in available_models][:1],
)

max_rows = st.number_input("Max rows used for training comparison", min_value=500, max_value=max(500, len(df)), value=min(2000, len(df)), step=250)
st.caption("Large datasets are sampled before model comparison to keep training responsive.")

if st.button("Train and compare models"):
    result = train_model(
        df,
        target_column,
        selected_models=selected_models,
        include_heavy_models=include_heavy_models,
        max_rows=int(max_rows),
    )
    if result is None:
        st.warning("A model could not be trained with the selected target.")
    else:
        st.session_state["last_model_report"] = result
        st.subheader("Model Comparison")
        st.dataframe(result["comparison"], use_container_width=True)
        st.subheader(f"Best Model: {result['best_model']}")
        st.json(result["best_metrics"])
        if result["feature_importance"]:
            st.subheader("Feature Importance")
            st.dataframe(result["feature_importance"], use_container_width=True)
