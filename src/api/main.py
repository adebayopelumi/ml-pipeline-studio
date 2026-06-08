import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.preprocess import load_preprocessor
from src.monitoring.logger import PredictionLogger
from src.core.algorithm_registry import ALGORITHM_REGISTRY, list_algorithms
from src.core.metrics import get_available_metrics

app = FastAPI(
    title="ML Pipeline API",
    version="3.0.0",
    description=(
        "REST API for the Production ML Pipeline. "
        "Train a model via the Streamlit app, then use these endpoints to run predictions "
        "and monitor drift from any external system."
    ),
)

MODEL_PATH        = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
LABEL_ENC_PATH    = Path("models/label_encoder.joblib")
STATE_PATH        = Path("models/last_run_state.json")

model         = None
preprocessor  = None
label_encoder = None
model_meta: Dict[str, Any] = {}
prediction_logger = PredictionLogger()


def _load_artifacts():
    global model, preprocessor, label_encoder, model_meta
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No trained model found at {MODEL_PATH}. Train a model via the app first.")
    model        = joblib.load(MODEL_PATH)
    preprocessor = load_preprocessor(str(PREPROCESSOR_PATH))
    label_encoder = joblib.load(LABEL_ENC_PATH) if LABEL_ENC_PATH.exists() else None
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        model_meta = {
            "algorithm":     state.get("algorithm", "unknown"),
            "problem_type":  state.get("problem_type", "unknown"),
            "target_column": state.get("target_column", "unknown"),
            "metrics":       state.get("metrics", {}),
        }


@app.on_event("startup")
def startup_event():
    try:
        _load_artifacts()
        print(f"Model loaded — algorithm={model_meta.get('algorithm')}, "
              f"target={model_meta.get('target_column')}")
    except RuntimeError as e:
        print(f"Warning: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "algorithm": model_meta.get("algorithm"),
        "target_column": model_meta.get("target_column"),
        "problem_type": model_meta.get("problem_type"),
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train one via the Streamlit app first.")
    return {
        "algorithm":     model_meta.get("algorithm"),
        "problem_type":  model_meta.get("problem_type"),
        "target_column": model_meta.get("target_column"),
        "training_metrics": model_meta.get("metrics"),
    }


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature values as a dict. Keys must match the columns used during training.",
        json_schema_extra={"example": {"Age": 35, "Fare": 52.5, "Pclass": 1, "Sex": "female"}},
    )


class PredictResponse(BaseModel):
    prediction: Any
    label: Optional[str]
    probability: Optional[float]
    problem_type: str


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dicts, one per row.",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    try:
        input_df  = pd.DataFrame([request.features])
        processed = preprocessor.transform(input_df)
        raw_pred  = model.predict(processed)[0]

        if label_encoder is not None:
            label = str(label_encoder.inverse_transform([int(raw_pred)])[0])
        else:
            label = str(raw_pred)

        prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(processed)[0]
            prob  = float(np.max(proba))

        prediction_logger.log_prediction(
            input_features=request.features,
            prediction=int(raw_pred) if isinstance(raw_pred, (int, np.integer)) else float(raw_pred),
            prediction_proba=prob or 0.0,
            model_version=model_meta.get("algorithm", "unknown"),
        )

        return PredictResponse(
            prediction=label,
            label=label,
            probability=prob,
            problem_type=model_meta.get("problem_type", "unknown"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    try:
        df        = pd.DataFrame(request.records)
        processed = preprocessor.transform(df)
        raw_preds = model.predict(processed)

        if label_encoder is not None:
            labels = label_encoder.inverse_transform(raw_preds.astype(int)).tolist()
        else:
            labels = [str(p) for p in raw_preds]

        results = [{"prediction": lbl} for lbl in labels]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(processed)
            for i, row in enumerate(results):
                row["probability"] = float(np.max(proba[i]))

        return {
            "count":   len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
def run_training(
    dataset_path:  str  = "data/raw/dataset.csv",
    target_column: str  = "target",
    algorithm:     str  = "auto",
    tune:          bool = False,
):
    """Train a model via the API. The result is saved to models/ and used by /predict."""
    from src.models.train import train as _train

    try:
        _model, _preprocessor, eval_metrics, problem_type, _eval_data = _train(
            dataset_path=dataset_path,
            target_column=target_column,
            algorithm=algorithm,
            tune=tune,
        )
        _load_artifacts()
        return {
            "status":       "success",
            "problem_type": problem_type,
            "algorithm":    algorithm,
            "metrics":      eval_metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/algorithms")
def algorithms(problem_type: str = None):
    names = list_algorithms(problem_type)
    return {
        "algorithms": {
            name: {
                "problem_type":   ALGORITHM_REGISTRY[name]["problem_type"],
                "hyperparameters": ALGORITHM_REGISTRY[name]["hyperparameters"],
            }
            for name in names
        }
    }


@app.get("/metrics")
def metrics(problem_type: str = None):
    if problem_type is None:
        return {
            "binary_classification": get_available_metrics("binary_classification"),
            "multiclass_classification": get_available_metrics("multiclass_classification"),
            "regression": get_available_metrics("regression"),
        }
    return {"metrics": get_available_metrics(problem_type)}
