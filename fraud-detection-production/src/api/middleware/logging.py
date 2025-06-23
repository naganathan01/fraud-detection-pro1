import time
import logging
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request
        logger = logging.getLogger("api.requests")
        logger.info(
            "Incoming request",
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'user_agent': request.headers.get('user-agent'),
                'user_id': getattr(request.state, 'user_id', 'anonymous')
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'process_time': round(process_time * 1000, 2),  # ms
                    'user_id': getattr(request.state, 'user_id', 'anonymous')
                }
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'process_time': round(process_time * 1000, 2),
                    'user_id': getattr(request.state, 'user_id', 'anonymous')
                }
            )
            raise