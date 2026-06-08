import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer


SAMPLE_FEATURES = {
    "Age": 35,
    "Fare": 52.5,
    "Pclass": 1,
    "Sex": "female",
}

SAMPLE_PAYLOAD = {"features": SAMPLE_FEATURES}


def make_mock_model():
    model = MagicMock(spec=RandomForestClassifier)
    model.predict.side_effect      = lambda X: np.ones(len(X), dtype=int)
    model.predict_proba.side_effect = lambda X: np.tile([0.3, 0.7], (len(X), 1))
    return model


def make_mock_preprocessor():
    preprocessor = MagicMock(spec=ColumnTransformer)
    preprocessor.transform.side_effect = lambda X: np.zeros((len(X), 10))
    return preprocessor


@pytest.fixture
def client():
    from src.api import main

    original_model        = main.model
    original_preprocessor = main.preprocessor

    # Set mocks before creating the client so startup doesn't overwrite them
    main.model        = make_mock_model()
    main.preprocessor = make_mock_preprocessor()

    # Not using context manager — avoids running @on_event("startup") which would
    # reload real artifacts from disk and overwrite the mocks.
    yield TestClient(main.app)

    main.model        = original_model
    main.preprocessor = original_preprocessor


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    # New API returns algorithm / problem_type / target_column
    assert "algorithm" in data or "problem_type" in data


def test_predict(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "problem_type" in data


def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={"records": [SAMPLE_FEATURES, SAMPLE_FEATURES]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
