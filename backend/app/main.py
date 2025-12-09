from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .config import settings
from .api.v1.api import router as api_router
from .models.ml_model import predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    predictor.load_models()
    yield
    print("Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dropout-teacher-portal.onrender.com",
        "https://dropout-frontend.onrender.com"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "message": "Dropout Prediction API",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }