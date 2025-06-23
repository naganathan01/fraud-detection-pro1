from fastapi import APIRouter, Depends
from src.api.schemas.response import HealthResponse
from src.utils.config import settings
from src.utils.cache import cache_manager
import redis
import psutil
import time

router = APIRouter()

@router.get("/live", response_model=dict)
async def liveness_check():
    """Basic liveness check - is the service running?"""
    return {"status": "alive", "timestamp": time.time()}

@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Comprehensive readiness check - is the service ready to serve traffic?"""
    
    dependencies = {}
    all_healthy = True
    
    # Check Redis connection
    try:
        if cache_manager.is_healthy():
            dependencies["redis"] = "healthy"
        else:
            dependencies["redis"] = "unhealthy"
            all_healthy = False
    except Exception as e:
        dependencies["redis"] = f"error: {str(e)}"
        all_healthy = False
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        dependencies["system"] = f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
        
        # Mark as unhealthy if resources are too high
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            all_healthy = False
            
    except Exception as e:
        dependencies["system"] = f"error: {str(e)}"
        all_healthy = False
    
    status = "ready" if all_healthy else "not_ready"
    
    return HealthResponse(
        status=status,
        environment=settings.ENVIRONMENT,
        dependencies=dependencies
    )

@router.get("/metrics")
async def get_health_metrics():
    """Get detailed health metrics"""
    
    try:
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
    except Exception as e:
        return {"error": str(e)}