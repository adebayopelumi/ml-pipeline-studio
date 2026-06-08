import math
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.utils.config import load_config
from src.data.load_data import load_raw_data, clean_raw_data
from src.data.validate_data import validate
from src.data.preprocess import preprocess, save_preprocessor
from src.core.problem_detector import detect_problem_type, describe_problem
from src.core.algorithm_registry import get_algorithm
from src.core.metrics import compute_metrics, confusion_matrix_report
from src.models.evaluate import get_feature_importances, find_optimal_threshold, save_metrics
from src.monitoring.logger import MetricsTracker


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)


def train(
    config: dict = None,
    dataset_path: str = None,
    target_column: str = None,
    algorithm: str = "auto",
    params: dict = None,
    metrics: list = None,
    tune: bool = False,
):
    """
    Train a model on any dataset.

    Args:
        config:        Config dict (loaded from yaml if not provided)
        dataset_path:  Path to CSV file (overrides config)
        target_column: Name of the target column (overrides config)
        algorithm:     Algorithm name or 'auto' to let the system pick
        params:        Manual hyperparameters (e.g. {"max_depth": 5})
        metrics:       List of metric names (e.g. ["f1", "roc_auc"]) or None for auto
        tune:          If True, run Optuna tuning before training
    """
    if config is None:
        config = load_config()

    raw_path = dataset_path or config["data"]["raw_path"]
    target   = target_column or config["data"]["target_column"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Load & prepare data ─────────────────────────────────────────────────
    df = load_raw_data(raw_path)
    df = clean_raw_data(df)
    df = validate(df, target_column=target)

    # ── Detect problem type ─────────────────────────────────────────────────
    problem_type = detect_problem_type(df[target])
    print(f"Problem type detected: {describe_problem(problem_type)}")

    # ── Resolve algorithm ───────────────────────────────────────────────────
    if tune:
        from src.models.tune import tune as run_tune
        best_params, _ = run_tune(
            config=config,
            dataset_path=raw_path,
            target_column=target,
            algorithm=algorithm,
            metrics=metrics,
        )
        params = best_params

    algo_entry = get_algorithm(algorithm, problem_type)
    model_params = {**algo_entry["default_params"], **(params or {})}
    model = algo_entry["model_class"](**model_params)

    # ── Preprocess ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess(
        df,
        target_column=target,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # ── Train & evaluate ────────────────────────────────────────────────────
    mlflow.end_run()  # close any orphaned run from a previous failed attempt
    with mlflow.start_run():
        mlflow.log_params(model_params)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("dataset", raw_path)

        model.fit(X_train, y_train)

        eval_metrics = compute_metrics(model, X_test, y_test, problem_type, metrics)
        # Filter out None / NaN values — MLflow rejects them and can corrupt the run
        loggable = {k: v for k, v in eval_metrics.items() if v is not None and not math.isnan(v)}
        mlflow.log_metrics(loggable)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="ml_model")

        print("\nEvaluation metrics:")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v:.4f}")

        extra = {}

        if "classification" in problem_type:
            cm = confusion_matrix_report(model, X_test, y_test)
            threshold_info = find_optimal_threshold(model, X_test, y_test)
            mlflow.log_metrics({f"cm_{k}": v for k, v in cm.items() if isinstance(v, (int, float))})
            if threshold_info["threshold"] is not None:
                mlflow.log_metric("optimal_threshold", threshold_info["threshold"])
            print(f"Confusion matrix: {cm}")
            t = threshold_info["threshold"]
            print(f"Optimal threshold: {f'{t:.2f}' if t else 'N/A (multiclass)'} (F1={threshold_info.get('f1_at_threshold', 'N/A')})")
            extra = {**cm, **threshold_info}

        feature_importances = get_feature_importances(model, preprocessor)
        if feature_importances:
            print("Top 5 features:", list(feature_importances.items())[:5])

        save_metrics({**eval_metrics, **extra, "feature_importances": feature_importances})
        MetricsTracker().log_metrics(eval_metrics, target_column=target, problem_type=problem_type)

    save_model(model, config["model"]["save_path"])
    save_preprocessor(preprocessor)
    if label_encoder is not None:
        joblib.dump(label_encoder, "models/label_encoder.joblib")
        print("Label encoder saved to models/label_encoder.joblib")

    # Build eval_data for Results page charts (stored in session state by caller)
    import numpy as np
    y_pred_arr = model.predict(X_test)
    y_prob_arr = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        y_prob_arr = proba.tolist()          # 2-D list works for both binary & multiclass

    # Densify X_test — ColumnTransformer may return a sparse matrix when
    # OneHotEncoder is present; SHAP and plotting code both need dense arrays.
    import scipy.sparse as sp
    X_test_dense = X_test.toarray() if sp.issparse(X_test) else np.array(X_test)

    eval_data = {
        "y_test":  y_test.tolist(),
        "y_pred":  y_pred_arr.tolist(),
        "y_prob":  y_prob_arr,
        "classes": label_encoder.classes_.tolist() if label_encoder is not None else None,
        "X_test":  X_test_dense,
    }

    return model, preprocessor, eval_metrics, problem_type, eval_data


if __name__ == "__main__":
    train()
