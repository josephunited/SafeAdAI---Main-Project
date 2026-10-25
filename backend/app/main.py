import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.db.database import engine, Base
from backend.app.api.v1 import auth, age, advertisements, moderation, admin

# Auto-create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="SafeAd AI (SAFE-VISION) Multimodal Trust & Safety Backend API",
    version="2.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Registrations
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(age.router, prefix=f"{settings.API_V1_STR}/age", tags=["Age Verification"])
app.include_router(advertisements.router, prefix=f"{settings.API_V1_STR}/advertisements", tags=["Advertisements"])
app.include_router(moderation.router, prefix=f"{settings.API_V1_STR}/moderation", tags=["Moderation & Audit"])
app.include_router(admin.router, prefix=f"{settings.API_V1_STR}/admin", tags=["Admin Control"])

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "database": settings.DATABASE_URL.split("://")[0]
    }
