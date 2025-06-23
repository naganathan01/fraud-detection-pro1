import jwt
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import time
from src.utils.config import settings

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests"""
    
    def __init__(self, app):
        super().__init__(app)
        self.security = HTTPBearer()
        
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoints and docs
        if request.url.path in ["/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Skip auth for metrics endpoint
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
            
        try:
            # Extract authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authorization header"
                )
            
            # Validate API key or JWT
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
                # Try API key first
                if token == settings.API_KEY:
                    request.state.user_id = "api_user"
                    request.state.authenticated = True
                else:
                    # Try JWT validation
                    try:
                        payload = jwt.decode(
                            token, 
                            settings.JWT_SECRET_KEY, 
                            algorithms=["HS256"]
                        )
                        request.state.user_id = payload.get("user_id", "unknown")
                        request.state.authenticated = True
                    except jwt.ExpiredSignatureError:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has expired"
                        )
                    except jwt.InvalidTokenError:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid token"
                        )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization format"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}"
            )
        
        response = await call_next(request)
        return response