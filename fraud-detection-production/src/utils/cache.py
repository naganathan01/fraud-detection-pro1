import redis
import json
import pickle
from typing import Any, Optional
import logging
from src.utils.config import settings

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=False,  # Keep as bytes for pickle
            health_check_interval=30,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
    
    async def get_cached_prediction(self, cache_key: str) -> Optional[dict]:
        """Get cached prediction result"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return None
    
    async def cache_prediction(self, cache_key: str, prediction_result: dict, ttl: int = 300):
        """Cache prediction result"""
        try:
            self.redis_client.setex(
                cache_key, 
                ttl, 
                pickle.dumps(prediction_result)
            )
        except Exception as e:
            logging.error(f"Cache set error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False

# Global cache instance
cache_manager = CacheManager()

async def get_cached_prediction(cache_key: str) -> Optional[dict]:
    return await cache_manager.get_cached_prediction(cache_key)

async def cache_prediction(cache_key: str, result: dict, ttl: int = 300):
    await cache_manager.cache_prediction(cache_key, result, ttl)