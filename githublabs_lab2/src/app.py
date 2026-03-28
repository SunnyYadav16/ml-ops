"""app.py — FastAPI inference endpoint for the Customer Churn XGBoost model.

Enhancement over original lab (which has no inference endpoint):
  - /health        → model version + readiness status
  - /predict       → single customer record → churn probability + prediction
  - /batch_predict → list of records → batch predictions
  - Loads the latest versioned model from models/ at startup

Run locally:
    uvicorn src.app:app --reload --port 8000

Then open: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import glob
import json
import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "XGBoost model trained on IBM Telco Customer Churn dataset. "
        "Model is trained and versioned automatically via GitHub Actions CI/CD."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Model state (loaded at startup)
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_pipeline = None
_model_version = None


def _load_latest_model():
    global _pipeline, _model_version

    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "model_*.joblib")))
    if not model_files:
        raise RuntimeError(f"No model files found in {MODEL_DIR}/")

    latest = model_files[-1]
    _pipeline = joblib.load(latest)
    _model_version = os.path.basename(latest).replace(".joblib", "")
    print(f"[app] loaded model: {_model_version}")


@app.on_event("startup")
def startup_event():
    try:
        _load_latest_model()
    except RuntimeError as e:
        print(f"[app] WARNING: {e} — starting in degraded mode")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CustomerRecord(BaseModel):
    """One Telco customer feature row. Defaults match a typical high-risk profile."""
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: float = 6
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 420.0


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int   # 1 = likely to churn, 0 = likely to stay
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if _pipeline is not None else "degraded",
        "model_version": _model_version,
        "model_loaded": _pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([record.dict()])
    prob = float(_pipeline.predict_proba(df)[:, 1][0])
    pred = int(prob >= 0.5)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        model_version=_model_version or "unknown",
    )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(records: List[CustomerRecord]):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not records:
        raise HTTPException(status_code=400, detail="Empty records list")

    df = pd.DataFrame([r.model_dump() for r in records])
    probs = _pipeline.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                churn_probability=round(float(p), 4),
                churn_prediction=int(c),
                model_version=_model_version or "unknown",
            )
            for p, c in zip(probs, preds)
        ],
        count=len(records),
    )
