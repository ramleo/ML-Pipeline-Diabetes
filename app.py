"""
Diabetes Prediction API
FastAPI application serving the LogisticRegression ML pipeline.
Version: 1.0.0
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/final_pipeline.pkl")
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
LABELS = {0: "No Diabetes", 1: "Diabetes"}
MODEL_NAME = "LogisticRegression"
API_VERSION = "1.0.0"

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Pipeline not found at {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Pipeline loaded successfully from %s", MODEL_PATH)
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Diabetes Prediction API",
    description="Binary classification: predicts whether a patient has diabetes.",
    version=API_VERSION,
    lifespan=lifespan,
)


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: Optional[float] = Field(None, ge=0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: Optional[float] = Field(None, ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: Optional[float] = Field(None, ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: Optional[float] = Field(None, ge=0, description="2-Hour serum insulin (mu U/ml)")
    BMI: Optional[float] = Field(None, ge=0, description="Body mass index (kg/m2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function score")
    Age: int = Field(..., ge=0, description="Age in years")


class PredictionResult(BaseModel):
    prediction: int
    label: str
    probability: Dict[str, float]


def _build_dataframe(inputs: List[DiabetesInput]) -> pd.DataFrame:
    rows = [
        {
            "Pregnancies": inp.Pregnancies,
            "Glucose": inp.Glucose,
            "BloodPressure": inp.BloodPressure,
            "SkinThickness": inp.SkinThickness,
            "Insulin": inp.Insulin,
            "BMI": inp.BMI,
            "DiabetesPedigreeFunction": inp.DiabetesPedigreeFunction,
            "Age": inp.Age,
        }
        for inp in inputs
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _predict(df: pd.DataFrame) -> List[PredictionResult]:
    try:
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append(
            PredictionResult(
                prediction=int(pred),
                label=LABELS[int(pred)],
                probability={
                    "no_diabetes": round(float(prob[0]), 4),
                    "diabetes": round(float(prob[1]), 4),
                },
            )
        )
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": API_VERSION}


@app.post("/predict", response_model=PredictionResult)
def predict(data: DiabetesInput):
    df = _build_dataframe([data])
    return _predict(df)[0]


class BatchRequest(BaseModel):
    records: List[DiabetesInput]


class BatchResponse(BaseModel):
    predictions: List[PredictionResult]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(data: BatchRequest):
    if not data.records:
        raise HTTPException(status_code=422, detail="Input list must not be empty.")
    df = _build_dataframe(data.records)
    return BatchResponse(predictions=_predict(df))
