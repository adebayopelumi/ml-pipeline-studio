import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess, build_preprocessor
from src.models.evaluate import compute_metrics
from sklearn.ensemble import RandomForestClassifier


def make_sample_df(n=100):
    rng = np.random.default_rng(42)
    contracts = ["Month-to-month", "One year", "Two year"]
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).astype(float),
        "total_charges": rng.uniform(0, 8000, n).astype(float),
        "contract": rng.choice(contracts, n),
        "churn": rng.integers(0, 2, n),
    })


def test_preprocess_returns_correct_shapes():
    df = make_sample_df()
    X_train, X_test, y_train, y_test, _ = preprocess(df, "churn", 0.2, 42)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]


def test_metrics_keys():
    df = make_sample_df()
    X_train, X_test, y_train, y_test, _ = preprocess(df, "churn", 0.2, 42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_metrics_values_in_range():
    df = make_sample_df()
    X_train, X_test, y_train, y_test, _ = preprocess(df, "churn", 0.2, 42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
