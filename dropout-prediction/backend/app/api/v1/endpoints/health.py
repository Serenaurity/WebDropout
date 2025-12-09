from fastapi import APIRouter
from ....models.schemas import HealthResponse
from ....models.ml_model import predictor

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health():
    loaded_terms = {k: v is not None for k, v in predictor.models.items()}
    loaded_count = sum(1 for v in loaded_terms.values() if v)
    return HealthResponse(
        status="healthy" if predictor.model_loaded else "unhealthy",
        model_loaded=predictor.model_loaded,
        loaded_terms=loaded_terms,
        loaded_count=loaded_count
    )

@router.options("/health")
async def health_options():
    return {"message": "OK"}