import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def make_mock_model():
    model = MagicMock(spec=RandomForestClassifier)
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


def make_mock_preprocessor():
    preprocessor = MagicMock(spec=ColumnTransformer)
    preprocessor.transform.return_value = np.zeros((1, 5))
    return preprocessor


@pytest.fixture
def client():
    from src.api import main

    # Save original values
    original_model = main.model
    original_preprocessor = main.preprocessor

    # Set mocks
    main.model = make_mock_model()
    main.preprocessor = make_mock_preprocessor()

    client = TestClient(main.app)

    yield client

    # Restore original values
    main.model = original_model
    main.preprocessor = original_preprocessor


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_type" in response.json()


def test_predict_churn(client):
    payload = {
        "tenure": 2,
        "monthly_charges": 99.0,
        "total_charges": 200.0,
        "contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["churn", "no churn"]
    assert "model_version" in data
    assert "probability" in data
