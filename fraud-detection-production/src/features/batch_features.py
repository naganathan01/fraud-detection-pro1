# src/features/batch_features.py
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta

class BatchFeatureEngine:
    """Batch feature processing for training and batch inference"""
    
    def __init__(self):
        self.feature_configs = self._load_feature_configs()
        
    def process_batch_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Process features for a batch of transactions"""
        
        logging.info(f"Processing batch features for {len(transactions_df)} transactions")
        
        # Create features
        features_df = transactions_df.copy()
        
        # Add user aggregation features
        features_df = self._add_user_aggregations(features_df)
        
        # Add merchant features
        features_df = self._add_merchant_features(features_df)
        
        # Add temporal features
        features_df = self._add_temporal_features(features_df)
        
        # Add risk scores
        features_df = self._add_risk_scores(features_df)
        
        logging.info(f"✅ Batch feature processing completed. Features: {features_df.shape[1]}")
        
        return features_df
    
    def _add_user_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user-level aggregation features"""
        
        # User transaction counts
        user_stats = df.groupby('user_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'transaction_id': 'count'
        }).fillna(0)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        
        # Merge back to main dataframe
        df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        return df
    
    def _add_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add merchant-level features"""
        
        # Merchant transaction patterns
        merchant_stats = df.groupby('merchant_id').agg({
            'amount': ['count', 'mean', 'std'],
            'is_fraud': 'mean'  # Fraud rate by merchant
        }).fillna(0)
        
        merchant_stats.columns = ['merchant_' + '_'.join(col).strip() for col in merchant_stats.columns]
        
        # Merge back
        df = df.merge(merchant_stats, left_on='merchant_id', right_index=True, how='left')
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if 'transaction_time' in df.columns:
            df['hour'] = pd.to_datetime(df['transaction_time']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['transaction_time']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        
        return df
    
    def _add_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-based features"""
        
        # Location risk (simplified)
        high_risk_countries = ['CN', 'RU', 'NG', 'PK']
        df['location_risk'] = df['location'].isin(high_risk_countries).astype(int)
        
        # Amount risk
        df['amount_risk'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        return df
    
    def _load_feature_configs(self) -> Dict:
        """Load feature configuration"""
        return {
            'numerical_features': [
                'amount', 'user_age', 'account_age_days', 'credit_score'
            ],
            'categorical_features': [
                'location', 'device_type', 'merchant_id'
            ],
            'derived_features': [
                'amount_log', 'velocity_score', 'risk_score'
            ]
        }