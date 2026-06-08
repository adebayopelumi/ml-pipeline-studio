import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess
from src.core.metrics import compute_metrics
from sklearn.ensemble import RandomForestClassifier


def make_sample_df(n=200):
    rng = np.random.default_rng(42)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    payment = [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    yes_no = ["Yes", "No"]

    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(yes_no, n),
        "Dependents": rng.choice(yes_no, n),
        "tenure": rng.integers(0, 72, n),
        "PhoneService": rng.choice(yes_no, n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(internet, n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(contracts, n),
        "PaperlessBilling": rng.choice(yes_no, n),
        "PaymentMethod": rng.choice(payment, n),
        "MonthlyCharges": rng.uniform(20, 120, n).astype(float),
        "TotalCharges": rng.uniform(0, 8000, n).astype(float),
        "Churn": rng.integers(0, 2, n),
    })


def test_preprocess_returns_correct_shapes():
    df = make_sample_df()
    X_train, X_test, y_train, y_test, _, _le = preprocess(df, "Churn", 0.2, 42)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]


def test_metrics_keys():
    df = make_sample_df()
    X_train, X_test, y_train, y_test, _, _le = preprocess(df, "Churn", 0.2, 42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test, "binary_classification")
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_metrics_values_in_range():
    df = make_sample_df()
    X_train, X_test, y_train, y_test, _, _le = preprocess(df, "Churn", 0.2, 42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test, "binary_classification")
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
