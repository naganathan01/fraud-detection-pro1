# src/features/real_time_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class RealTimeFeatureEngine:
    def __init__(self, feature_store):
        self.feature_store = feature_store
        
    def extract_features(self, transaction: Dict, user_id: str) -> Dict:
        """Extract all features for fraud prediction"""
        
        # Get cached user features
        user_features = self.feature_store.get_user_features(user_id)
        
        # Transaction-level features
        txn_features = self._extract_transaction_features(transaction)
        
        # Velocity features
        velocity_features = self._extract_velocity_features(transaction, user_features)
        
        # Behavioral features
        behavioral_features = self._extract_behavioral_features(transaction, user_features)
        
        # Risk features
        risk_features = self._extract_risk_features(transaction)
        
        # Combine all features
        all_features = {
            **txn_features,
            **velocity_features,
            **behavioral_features,
            **risk_features
        }
        
        return all_features
    
    def _extract_transaction_features(self, transaction: Dict) -> Dict:
        """Basic transaction features"""
        amount = transaction['amount']
        
        return {
            'amount': amount,
            'amount_log': np.log1p(amount),
            'amount_rounded': amount % 1 == 0,  # Round number indicator
            'is_weekend': datetime.now().weekday() >= 5,
            'hour_of_day': datetime.now().hour,
            'is_business_hours': 9 <= datetime.now().hour <= 17
        }
    
    def _extract_velocity_features(self, transaction: Dict, user_features: Dict) -> Dict:
        """Velocity-based features"""
        amount = transaction['amount']
        avg_amount = user_features.get('avg_amount_30d', 0)
        
        return {
            'txn_count_1h': user_features.get('txn_count_1h', 0),
            'txn_count_24h': user_features.get('txn_count_24h', 0),
            'amount_zscore': (amount - avg_amount) / max(avg_amount * 0.1, 1),
            'velocity_score': user_features.get('velocity_score', 0)
        }
    
    def _extract_behavioral_features(self, transaction: Dict, user_features: Dict) -> Dict:
        """Behavioral pattern features"""
        return {
            'is_foreign_transaction': transaction.get('is_foreign_transaction', False),
            'device_type_encoded': self._encode_device_type(transaction.get('device_type', 'unknown')),
            'location_risk': self._get_location_risk(transaction.get('location', 'unknown'))
        }
    
    def _extract_risk_features(self, transaction: Dict) -> Dict:
        """Risk-based features"""
        return {
            'is_high_risk_country': transaction.get('is_high_risk_country', False),
            'merchant_risk_score': self._get_merchant_risk(transaction.get('merchant_id', 'unknown')),
            'time_since_last_txn': self._get_time_since_last_transaction(transaction)
        }
    
    def _encode_device_type(self, device_type: str) -> int:
        """Encode device type"""
        mapping = {'mobile': 0, 'desktop': 1, 'tablet': 2}
        return mapping.get(device_type.lower(), 3)
    
    def _get_location_risk(self, location: str) -> float:
        """Get location-based risk score"""
        # Simplified - in production, use ML model or lookup table
        high_risk_locations = ['location_1', 'location_3']
        return 0.8 if location in high_risk_locations else 0.2
    
    def _get_merchant_risk(self, merchant_id: str) -> float:
        """Get merchant risk score"""
        # In production, maintain merchant risk database
        return 0.5  # Default medium risk
    
    def _get_time_since_last_transaction(self, transaction: Dict) -> float:
        """Calculate time since last transaction in hours"""
        # In production, get from user history
        return 24.0  # Default 24 hours