from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None


FAST_CLASSIFICATION_MODELS = ["Logistic Regression", "Decision Tree"]
FAST_REGRESSION_MODELS = ["Linear Regression", "Ridge", "Decision Tree"]


def infer_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )


def get_candidate_models(problem_type: str, include_heavy_models: bool = False) -> dict[str, object]:
    if problem_type == "regression":
        models: dict[str, object] = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }
        if include_heavy_models:
            models["Random Forest"] = RandomForestRegressor(random_state=42)
            models["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
        return models

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    if include_heavy_models:
        models["Random Forest"] = RandomForestClassifier(random_state=42)
        models["KNN"] = KNeighborsClassifier()
        models["SVM"] = SVC(probability=True)
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss")
    return models


def _safe_roc_auc(y_true: pd.Series, predictions: pd.Series, probabilities: pd.DataFrame | None = None) -> float | None:
    try:
        if probabilities is not None:
            if len(pd.unique(y_true)) == 2:
                return float(roc_auc_score(y_true, probabilities[:, 1]))
            return float(roc_auc_score(y_true, probabilities, multi_class="ovr"))
        return float(roc_auc_score(y_true, predictions))
    except Exception:
        return None


def _feature_importance(pipeline: Pipeline, X: pd.DataFrame) -> list[dict]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    ordered = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(value)} for name, value in ordered[:10]]


def _score_prediction(problem_type: str, y_test: pd.Series, predictions: pd.Series, probabilities: pd.DataFrame | None = None) -> tuple[dict[str, object], float]:
    if problem_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "roc_auc": _safe_roc_auc(y_test, predictions, probabilities),
        }
        score = metrics["f1"] if metrics["f1"] is not None else metrics["accuracy"]
        return metrics, float(score)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(mse) ** 0.5, 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
    return metrics, float(metrics["r2"])


def _fit_and_score(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
) -> tuple[dict[str, object], float, Pipeline] | None:
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except Exception:
        return None

    probabilities = None
    if problem_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X_test)
        except Exception:
            probabilities = None

    metrics, score = _score_prediction(problem_type, y_test, predictions, probabilities)
    metrics = {"model": model_name, **metrics}
    return metrics, score, pipeline


def train_model(
    df: pd.DataFrame,
    target_column: str,
    selected_models: list[str] | None = None,
    include_heavy_models: bool = False,
    max_rows: int = 5000,
) -> dict | None:
    if df is None or df.empty or target_column not in df.columns:
        return None

    data = df.dropna(subset=[target_column]).copy()
    if data.shape[0] < 10:
        return None

    y = data[target_column]
    X = data.drop(columns=[target_column])
    if X.empty:
        return None

    if len(data) > max_rows:
        sampled = data.sample(n=max_rows, random_state=42)
        y = sampled[target_column]
        X = sampled.drop(columns=[target_column])
        data = sampled

    problem_type = infer_problem_type(y)
    preprocessor = build_preprocessor(X)
    models = get_candidate_models(problem_type, include_heavy_models=include_heavy_models)
    if selected_models:
        models = {name: model for name, model in models.items() if name in selected_models}
    if not models:
        return None

    if len(data) < 20:
        return None

    stratify_target = None
    if problem_type == "classification":
        class_counts = y.value_counts(dropna=False)
        if len(class_counts) > 1 and int(class_counts.min()) >= 2:
            stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    comparison_rows = []
    best_result: dict | None = None

    for model_name, model in models.items():
        scored = _fit_and_score(model_name, model, X_train, X_test, y_train, y_test, preprocessor, problem_type)
        if scored is None:
            continue
        row, score, pipeline = scored

        comparison_rows.append(row)
        if best_result is None or score > best_result["score"]:
            best_result = {
                "score": float(score),
                "model_name": model_name,
                "pipeline": pipeline,
                "metrics": row,
            }

    if not comparison_rows or best_result is None:
        fallback_model = DummyClassifier(strategy="most_frequent") if problem_type == "classification" else DummyRegressor(strategy="mean")
        fallback = _fit_and_score("Fallback Baseline", fallback_model, X_train, X_test, y_train, y_test, preprocessor, problem_type)
        if fallback is None:
            return None
        row, score, best_pipeline = fallback
        row["model"] = "Fallback Baseline"
        return {
            "task": problem_type,
            "rows_used": int(len(data)),
            "comparison": [row],
            "best_model": "Fallback Baseline",
            "best_metrics": row,
            "feature_importance": _feature_importance(best_pipeline, X),
        }

    best_pipeline: Pipeline = best_result["pipeline"]
    result = {
        "task": problem_type,
        "rows_used": int(len(data)),
        "comparison": comparison_rows,
        "best_model": best_result["model_name"],
        "best_metrics": best_result["metrics"],
        "feature_importance": _feature_importance(best_pipeline, X),
    }
    return result
