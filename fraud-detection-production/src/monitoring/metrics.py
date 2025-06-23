# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import logging
from typing import Dict, Optional

# Request metrics
request_count = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'fraud_api_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

# Model metrics
model_predictions = Counter(
    'fraud_model_predictions_total',
    'Total model predictions',
    ['model_version', 'result']
)

model_latency = Histogram(
    'fraud_model_latency_seconds',
    'Model prediction latency',
    ['model_name']
)

fraud_detection_rate = Gauge(
    'fraud_detection_rate',
    'Current fraud detection rate'
)

# Business metrics
transactions_processed = Counter(
    'transactions_processed_total',
    'Total transactions processed',
    ['result']
)

fraud_amount_prevented = Counter(
    'fraud_amount_prevented_total',
    'Total fraud amount prevented in USD'
)

# Feature store metrics
feature_cache_hits = Counter(
    'feature_cache_hits_total',
    'Feature store cache hits'
)

feature_cache_misses = Counter(
    'feature_cache_misses_total',  
    'Feature store cache misses'
)

class ModelMetrics:
    """Model-specific metrics collection"""
    
    def __init__(self):
        self.prediction_times = []
        self.fraud_predictions = 0
        self.total_predictions = 0
        
    def record_prediction(self, latency: float, probability: float, is_fraud: bool):
        """Record prediction metrics"""
        # Update counters
        self.total_predictions += 1
        if is_fraud:
            self.fraud_predictions += 1
            
        # Record latency
        model_latency.labels(model_name='ensemble').observe(latency)
        
        # Record prediction
        result = 'fraud' if is_fraud else 'legitimate'
        model_predictions.labels(model_version='v2.0', result=result).inc()
        transactions_processed.labels(result=result).inc()
        
        # Update fraud detection rate
        if self.total_predictions > 0:
            rate = self.fraud_predictions / self.total_predictions
            fraud_detection_rate.set(rate)
            
    def record_error(self, error_type: str):
        """Record prediction errors"""
        model_predictions.labels(model_version='v2.0', result='error').inc()
        logging.error(f"Model prediction error: {error_type}")
        
    def record_business_impact(self, transaction_amount: float, is_fraud_prevented: bool):
        """Record business impact metrics"""
        if is_fraud_prevented:
            fraud_amount_prevented.inc(transaction_amount)