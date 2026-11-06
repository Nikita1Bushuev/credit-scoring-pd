from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Credit Default Prediction API",
    description="API для предсказания вероятности дефолта (PD-модель)",
    version="1.0.0",
)

app.include_router(router)
