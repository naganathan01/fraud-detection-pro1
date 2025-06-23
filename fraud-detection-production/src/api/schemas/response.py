from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool = Field(..., description="Whether transaction is predicted as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    risk_score: str = Field(..., description="Risk category (very_low, low, medium, high, very_high)")
    explanation: Dict[str, Any] = Field(..., description="Prediction explanation")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    features_used: int = Field(..., description="Number of features used in prediction")
    model_version: Optional[str] = Field(None, description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    errors: List[Dict[str, str]] = Field(default_factory=list)
    total_processed: int
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "2.0.0"
    environment: str
    dependencies: Dict[str, str]

class ModelStatusResponse(BaseModel):
    model_name: str
    version: str
    status: str
    accuracy: Optional[float] = None
    last_training: Optional[datetime] = None
    predictions_served: int