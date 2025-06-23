import time
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis
from src.utils.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis"""
    
    def __init__(self, app):
        super().__init__(app)
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        self.rate_limit = settings.RATE_LIMIT_PER_MINUTE
        
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
            
        # Get client identifier
        client_ip = self._get_client_ip(request)
        user_id = getattr(request.state, 'user_id', 'anonymous')
        rate_limit_key = f"rate_limit:{client_ip}:{user_id}"
        
        try:
            # Check current request count
            current_requests = self.redis_client.get(rate_limit_key)
            
            if current_requests is None:
                # First request in this minute
                self.redis_client.setex(rate_limit_key, 60, 1)
            else:
                current_requests = int(current_requests)
                if current_requests >= self.rate_limit:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Maximum {self.rate_limit} requests per minute."
                    )
                else:
                    # Increment counter
                    self.redis_client.incr(rate_limit_key)
                    
        except HTTPException:
            raise
        except Exception as e:
            # If Redis fails, allow the request but log the error
            print(f"Rate limiting error: {e}")
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.rate_limit - int(current_requests or 0))
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded IP (load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host