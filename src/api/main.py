import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from src.utils.config import load_config
from src.data.preprocess import load_preprocessor
from src.monitoring.logger import PredictionLogger

app = FastAPI(title="Churn Prediction API", version="1.0.0")

config = load_config()
MODEL_PATH = config["model"]["save_path"]
PREPROCESSOR_PATH = "models/preprocessor.joblib"

model = None
preprocessor = None
prediction_logger = PredictionLogger()


def load_artifacts():
    global model, preprocessor
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)


@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
        print("Model and preprocessor loaded successfully.")
    except RuntimeError as e:
        print(f"Warning: {e}")


class PredictRequest(BaseModel):
    tenure: int = Field(..., ge=0, description="Months as a customer")
    monthly_charges: float = Field(..., gt=0)
    total_charges: float = Field(..., ge=0)
    contract: str = Field(..., description="Month-to-month, One year, or Two year")


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": config["model"]["type"],
        "model_path": MODEL_PATH,
        "parameters": config["model"]["parameters"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    input_data = {
        "tenure": request.tenure,
        "monthly_charges": request.monthly_charges,
        "total_charges": request.total_charges,
        "contract": request.contract,
    }
    input_df = pd.DataFrame([input_data])

    processed = preprocessor.transform(input_df)
    prediction = int(model.predict(processed)[0])
    probability = float(model.predict_proba(processed)[0][1])
    label = "churn" if prediction == 1 else "no churn"

    # Log prediction for monitoring
    prediction_logger.log_prediction(
        input_features=input_data,
        prediction=prediction,
        prediction_proba=probability,
        model_version="v1"
    )

    return PredictResponse(
        prediction=prediction,
        prediction_label=label,
        probability=probability,
        model_version="v1",
    )
