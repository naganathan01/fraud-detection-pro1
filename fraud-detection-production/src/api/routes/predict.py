# src/api/routes/predict.py
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import time
import asyncio
from typing import Dict

from src.api.schemas.request import TransactionRequest, BatchTransactionRequest
from src.api.schemas.response import PredictionResponse, BatchPredictionResponse
from src.models.prediction import FraudPredictionService
from src.monitoring.metrics import request_count, request_duration
from src.utils.cache import cache_prediction, get_cached_prediction

router = APIRouter()
security = HTTPBearer()

# Initialize prediction service (singleton)
prediction_service = FraudPredictionService()

@router.post("/predict", 
             response_model=PredictionResponse,
             summary="Predict fraud for single transaction")
async def predict_fraud(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Predict fraud probability for a single transaction
    
    - **transaction_id**: Unique transaction identifier
    - **amount**: Transaction amount
    - **user_id**: User identifier
    - **merchant_id**: Merchant identifier
    - **location**: Transaction location
    - **device_type**: Device used for transaction
    """
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"prediction:{request.transaction_id}"
        cached_result = await get_cached_prediction(cache_key)
        
        if cached_result:
            request_count.labels(method="POST", endpoint="/predict", status="200").inc()
            return cached_result
        
        # Make prediction
        transaction_dict = request.dict()
        result = prediction_service.predict_fraud(transaction_dict)
        
        # Cache result for 5 minutes
        background_tasks.add_task(cache_prediction, cache_key, result, 300)
        
        # Record metrics
        duration = time.time() - start_time
        request_duration.labels(method="POST", endpoint="/predict").observe(duration)
        request_count.labels(method="POST", endpoint="/predict", status="200").inc()
        
        return PredictionResponse(**result)
        
    except Exception as e:
        request_count.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch",
             response_model=BatchPredictionResponse,
             summary="Predict fraud for multiple transactions")
async def predict_fraud_batch(
    request: BatchTransactionRequest,
    background_tasks: BackgroundTasks,
    auth: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Predict fraud probability for multiple transactions (up to 100)
    """
    if len(request.transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions per batch")
    
    start_time = time.time()
    
    try:
        # Process transactions concurrently
        tasks = []
        for transaction in request.transactions:
            task = asyncio.create_task(
                process_single_transaction(transaction.dict())
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        predictions = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    'transaction_id': request.transactions[i].transaction_id,
                    'error': str(result)
                })
            else:
                predictions.append(result)
        
        # Record metrics
        duration = time.time() - start_time
        request_duration.labels(method="POST", endpoint="/predict/batch").observe(duration)
        request_count.labels(method="POST", endpoint="/predict/batch", status="200").inc()
        
        return BatchPredictionResponse(
            predictions=predictions,
            errors=errors,
            total_processed=len(predictions),
            processing_time_ms=round(duration * 1000, 2)
        )
        
    except Exception as e:
        request_count.labels(method="POST", endpoint="/predict/batch", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

async def process_single_transaction(transaction: Dict) -> Dict:
    """Process single transaction asynchronously"""
    return prediction_service.predict_fraud(transaction)