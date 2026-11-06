from fastapi import APIRouter

from src.api.schemas import ClientData, PredictionResponse
from src.api.services import model_service

router = APIRouter()


@router.get("/")
def read_root():
    return {"message": "Credit Default Prediction API is alive!"}


@router.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_service.model is not None}


@router.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    return model_service.predict(data.model_dump())
