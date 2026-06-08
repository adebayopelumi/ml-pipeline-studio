import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
)


def _safe_roc_auc(y_true, y_prob):
    """Return None if roc_auc can't be computed (e.g. only one class in test set)."""
    if y_prob is None:
        return None
    try:
        if y_prob.ndim == 2:
            return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return None


CLASSIFICATION_METRICS = {
    "accuracy":  lambda y, yp, yb: accuracy_score(y, yp),
    "precision": lambda y, yp, yb: precision_score(y, yp, average="weighted", zero_division=0),
    "recall":    lambda y, yp, yb: recall_score(y, yp, average="weighted", zero_division=0),
    "f1":        lambda y, yp, yb: f1_score(y, yp, average="weighted", zero_division=0),
    "roc_auc":   lambda y, yp, yb: _safe_roc_auc(y, yb),
}

REGRESSION_METRICS = {
    "rmse": lambda y, yp: float(np.sqrt(mean_squared_error(y, yp))),
    "mae":  lambda y, yp: float(mean_absolute_error(y, yp)),
    "r2":   lambda y, yp: float(r2_score(y, yp)),
    "mape": lambda y, yp: float(np.mean(np.abs((y - yp) / np.where(y == 0, 1, y))) * 100),
}

DEFAULT_METRICS = {
    "binary_classification":     ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "multiclass_classification": ["accuracy", "precision", "recall", "f1"],
    "regression":                ["rmse", "mae", "r2"],
}


def compute_metrics(model, X_test, y_test, problem_type: str, selected_metrics: list = None) -> dict:
    is_classification = "classification" in problem_type
    metrics_to_use = selected_metrics or DEFAULT_METRICS[problem_type]
    results = {}

    if is_classification:
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            y_prob = proba[:, 1] if proba.shape[1] == 2 else proba

        for metric in metrics_to_use:
            if metric not in CLASSIFICATION_METRICS:
                raise ValueError(f"Unknown classification metric '{metric}'. Available: {list(CLASSIFICATION_METRICS.keys())}")
            value = CLASSIFICATION_METRICS[metric](y_test, y_pred, y_prob)
            if value is not None:
                results[metric] = float(value)
    else:
        y_pred = model.predict(X_test)
        for metric in metrics_to_use:
            if metric not in REGRESSION_METRICS:
                raise ValueError(f"Unknown regression metric '{metric}'. Available: {list(REGRESSION_METRICS.keys())}")
            results[metric] = REGRESSION_METRICS[metric](np.array(y_test), y_pred)

    return results


def confusion_matrix_report(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {"matrix": cm.tolist()}


def get_available_metrics(problem_type: str) -> list:
    if "classification" in problem_type:
        return list(CLASSIFICATION_METRICS.keys())
    return list(REGRESSION_METRICS.keys())
