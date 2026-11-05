from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.build_features import build_features

app = FastAPI(
    title="Credit Default Prediction API",
    description="API для предсказания вероятности дефолта (PD-модель)",
    version="1.0.0",
)

MODEL_PATH = Path("models/model_gbc_th0.3.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None


class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/")
def read_root():
    return {"message": "Credit Default Prediction API is alive!"}


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: ClientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    input_df = pd.DataFrame([data.model_dump()])
    try:
        processed_df = build_features(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")
    try:
        proba = model.predict_proba(processed_df)[0, 1]
        prediction = int(proba >= 0.3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "default_prediction": prediction,
        "default_probability": float(proba),
        "threshold_used": 0.3,
    }
