# src/models/prediction.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import time
import logging
from prometheus_client import Counter, Histogram, Gauge

from src.features.feature_store import FeatureStore
from src.features.real_time_features import RealTimeFeatureEngine
from src.models.ensemble import FraudDetectionEnsemble
from src.monitoring.metrics import ModelMetrics

class FraudPredictionService:
    def __init__(self):
        # Initialize components
        self.feature_store = FeatureStore()
        self.feature_engine = RealTimeFeatureEngine(self.feature_store)
        
        # Load model ensemble
        model_paths = {
            'xgboost': 'models/xgboost_fraud_model.pkl',
            'lightgbm': 'models/lightgbm_fraud_model.pkl',
            'neural_net': 'models/neural_net_fraud_model.pkl'
        }
        self.ensemble = FraudDetectionEnsemble(model_paths)
        
        # Initialize metrics
        self.metrics = ModelMetrics()
        
        # Circuit breaker state
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
    def predict_fraud(self, transaction: Dict) -> Dict:
        """Main prediction function with monitoring"""
        start_time = time.time()
        
        try:
            # Extract user ID
            user_id = transaction.get('user_id', 'anonymous')
            
            # Feature engineering
            features = self.feature_engine.extract_features(transaction, user_id)
            features_df = pd.DataFrame([features])
            
            # Make prediction
            with self.circuit_breaker:
                is_fraud, probability, explanation = self.ensemble.predict(features_df)
            
            # Update user features asynchronously
            self.feature_store.update_user_features(user_id, transaction)
            
            # Record metrics
            prediction_time = time.time() - start_time
            self.metrics.record_prediction(prediction_time, probability, is_fraud)
            
            # Prepare response
            response = {
                'transaction_id': transaction.get('transaction_id'),
                'is_fraud': is_fraud,
                'fraud_probability': probability,
                'risk_score': self._calculate_risk_score(probability),
                'explanation': explanation,
                'latency_ms': round(prediction_time * 1000, 2),
                'features_used': len(features)
            }
            
            return response
            
        except Exception as e:
            # Record error and return safe response
            self.metrics.record_error(str(e))
            logging.error(f"Prediction error: {e}")
            
            return {
                'transaction_id': transaction.get('transaction_id'),
                'is_fraud': False,  # Fail safe - don't block transactions
                'fraud_probability': 0.5,
                'risk_score': 'medium',
                'explanation': {'error': 'prediction_service_error'},
                'latency_ms': round((time.time() - start_time) * 1000, 2),
                'features_used': 0
            }
    
    def _calculate_risk_score(self, probability: float) -> str:
        """Convert probability to risk category"""
        if probability >= 0.8:
            return 'very_high'
        elif probability >= 0.6:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        elif probability >= 0.2:
            return 'low'
        else:
            return 'very_low'

class CircuitBreaker:
    """Circuit breaker for model predictions"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __enter__(self):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
        else:
            self.failure_count = 0
            self.state = 'CLOSED'