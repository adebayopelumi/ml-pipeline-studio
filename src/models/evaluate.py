import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

# compute_metrics and confusion_matrix_report now live in src/core/metrics.py
# This module keeps the utility functions used by train.py


def get_feature_importances(model, preprocessor) -> dict:
    if not hasattr(model, "feature_importances_"):
        return {}
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    return dict(
        sorted(
            zip(feature_names.tolist(), model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )


def find_optimal_threshold(model, X_test, y_test) -> dict:
    """Only meaningful for binary classification — skip for multiclass/regression."""
    if not hasattr(model, "predict_proba"):
        return {"threshold": 0.5, "f1_at_threshold": None}
    proba = model.predict_proba(X_test)
    if proba.shape[1] != 2:
        # Multiclass: threshold optimisation doesn't apply
        return {"threshold": None, "f1_at_threshold": None}
    y_prob = proba[:, 1]
    best_threshold, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        score = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, float(t)
    return {"threshold": best_threshold, "f1_at_threshold": best_f1}


def save_metrics(metrics: dict, path: str = "logs/metrics/eval_metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {path}")
