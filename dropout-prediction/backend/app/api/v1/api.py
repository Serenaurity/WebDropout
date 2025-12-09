from fastapi import APIRouter
from .endpoints import health, prediction, batch

router = APIRouter()
router.include_router(health.router, tags=["Health"])
router.include_router(prediction.router, tags=["Prediction"])
router.include_router(batch.router, tags=["Batch"])
