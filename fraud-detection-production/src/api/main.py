# src/api/main.py
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import time
import logging
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import predict, health, admin
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.logging import LoggingMiddleware
from src.utils.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Fraud Detection API",
    description="Production-grade real-time fraud detection service",
    version="2.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None
)

# Add middleware (order matters!)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, 
                  allow_origins=settings.ALLOWED_ORIGINS,
                  allow_methods=["GET", "POST"],
                  allow_headers=["*"])

# Custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(predict.router, prefix="/api/v1")
app.include_router(health.router, prefix="/health")
app.include_router(admin.router, prefix="/admin")

# Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logging.info("🚀 Starting Fraud Detection API v2.0.0")
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("🛑 Shutting down Fraud Detection API")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        workers=settings.WORKERS,
        loop="asyncio",
        access_log=True
    )