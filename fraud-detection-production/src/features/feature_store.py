# src/features/feature_store.py
import redis
import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class FeatureStore:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True,
            health_check_interval=30
        )
        
    def get_user_features(self, user_id: str) -> Dict:
        """Get real-time user features with <10ms latency"""
        try:
            # Get cached features
            cached_features = self.redis_client.hgetall(f"user_features:{user_id}")
            
            if not cached_features:
                return self._compute_default_features()
                
            return {
                'txn_count_1h': int(cached_features.get('txn_count_1h', 0)),
                'txn_count_24h': int(cached_features.get('txn_count_24h', 0)),
                'avg_amount_30d': float(cached_features.get('avg_amount_30d', 0)),
                'last_transaction_time': cached_features.get('last_transaction_time'),
                'velocity_score': float(cached_features.get('velocity_score', 0)),
                'risk_score': float(cached_features.get('risk_score', 0.5))
            }
        except Exception as e:
            # Fallback to default features
            return self._compute_default_features()
    
    def update_user_features(self, user_id: str, transaction: Dict):
        """Update features in real-time after transaction"""
        pipe = self.redis_client.pipeline()
        
        # Increment counters
        pipe.hincrby(f"user_features:{user_id}", "txn_count_1h", 1)
        pipe.hincrby(f"user_features:{user_id}", "txn_count_24h", 1)
        
        # Update last transaction
        pipe.hset(f"user_features:{user_id}", "last_transaction_time", 
                 datetime.now().isoformat())
        
        # Set expiry
        pipe.expire(f"user_features:{user_id}", 86400)  # 24 hours
        
        pipe.execute()
    
    def _compute_default_features(self) -> Dict:
        """Default features for new users"""
        return {
            'txn_count_1h': 0,
            'txn_count_24h': 0,
            'avg_amount_30d': 0.0,
            'last_transaction_time': None,
            'velocity_score': 0.0,
            'risk_score': 0.5
        }