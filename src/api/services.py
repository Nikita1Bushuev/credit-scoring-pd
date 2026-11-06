from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

from src.features.build_features import build_features

MODEL_PATH = Path("models/model_gbc_th0.3.pkl")
THRESHOLD = 0.3


class ModelService:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, data_dict: dict) -> dict:
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        df = pd.DataFrame([data_dict])
        try:
            processed_df = build_features(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"FE failed: {str(e)}")

        try:
            proba = self.model.predict_proba(processed_df)[0, 1]
            prediction = int(proba >= THRESHOLD)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        return {
            "default_prediction": prediction,
            "default_probability": float(proba),
            "threshold_used": THRESHOLD,
        }


model_service = ModelService()
