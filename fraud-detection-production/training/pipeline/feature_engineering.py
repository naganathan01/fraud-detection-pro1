import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import logging
from typing import Dict, List

class FeatureEngineer:
    """Feature engineering for fraud detection model"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for model training"""
        
        logging.info("Starting feature engineering...")
        
        # Create copy to avoid modifying original
        features_df = df.copy()
        
        # Amount-based features
        features_df = self._create_amount_features(features_df)
        
        # Time-based features
        features_df = self._create_time_features(features_df)
        
        # User-based features
        features_df = self._create_user_features(features_df)
        
        # Velocity features
        features_df = self._create_velocity_features(features_df)
        
        # Risk features
        features_df = self._create_risk_features(features_df)
        
        # Encode categorical features
        features_df = self._encode_categorical_features(features_df)
        
        # Scale numerical features
        features_df = self._scale_numerical_features(features_df)
        
        logging.info(f"✅ Feature engineering completed. Shape: {features_df.shape}")
        
        return features_df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features"""
        
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_rounded'] = (df['amount'] % 1 == 0).astype(int)
        df['amount_sqrt'] = np.sqrt(df['amount'])
        
        # Amount percentile within user
        df['amount_user_percentile'] = df.groupby('user_id')['amount'].rank(pct=True)
        
        # Amount Z-score within user
        user_amount_stats = df.groupby('user_id')['amount'].agg(['mean', 'std'])
        df = df.merge(user_amount_stats, left_on='user_id', right_index=True, suffixes=('', '_user'))
        df['amount_user_zscore'] = (df['amount'] - df['mean']) / (df['std'] + 1e-8)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Business hours
        df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_late_night'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        
        return df
    
    def _create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-based features"""
        
        # User age groups
        df['user_age_group'] = pd.cut(
            df['user_age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['young', 'adult', 'middle', 'senior', 'elderly']
        )
        
        # Account age groups
        df['account_age_group'] = pd.cut(
            df['account_age_days'],
            bins=[0, 30, 180, 365, 1000, 10000],
            labels=['new', 'recent', 'established', 'mature', 'veteran']
        )
        
        # Credit score categories
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['poor', 'fair', 'good', 'very_good', 'excellent']
        )
        
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity-based features"""
        
        # Sort by user and time for velocity calculations
        df = df.sort_values(['user_id', 'transaction_time'])
        
        # Transaction frequency features
        df['txn_count_last_1h'] = df.groupby('user_id').apply(
            lambda x: x.rolling('1H', on='transaction_time')['transaction_id'].count()
        ).reset_index(level=0, drop=True)
        
        df['txn_count_last_24h'] = df.groupby('user_id').apply(
            lambda x: x.rolling('24H', on='transaction_time')['transaction_id'].count()
        ).reset_index(level=0, drop=True)
        
        # Amount velocity
        df['amount_sum_last_1h'] = df.groupby('user_id').apply(
            lambda x: x.rolling('1H', on='transaction_time')['amount'].sum()
        ).reset_index(level=0, drop=True)
        
        df['amount_sum_last_24h'] = df.groupby('user_id').apply(
            lambda x: x.rolling('24H', on='transaction_time')['amount'].sum()
        ).reset_index(level=0, drop=True)
        
        # Time since last transaction
        df['hours_since_last_txn'] = df['hours_since_last_txn'].fillna(24.0)
        df['velocity_score'] = 1 / (df['hours_since_last_txn'] + 0.1)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-based features"""
        
        # Location risk
        location_fraud_rate = df.groupby('location')['is_fraud'].mean()
        df['location_risk_score'] = df['location'].map(location_fraud_rate)
        
        # Merchant risk
        merchant_fraud_rate = df.groupby('merchant_id')['is_fraud'].mean()
        df['merchant_risk_score'] = df['merchant_id'].map(merchant_fraud_rate)
        
        # Device risk
        device_fraud_rate = df.groupby('device_type')['is_fraud'].mean()
        df['device_risk_score'] = df['device_type'].map(device_fraud_rate)
        
        # Combined risk score
        df['combined_risk_score'] = (
            df['location_risk_score'].fillna(0.5) * 0.3 +
            df['merchant_risk_score'].fillna(0.5) * 0.4 +
            df['device_risk_score'].fillna(0.5) * 0.3
        )
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        
        categorical_features = [
            'location', 'device_type', 'user_age_group', 
            'account_age_group', 'credit_score_category'
        ]
        
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    df[f'{feature}_encoded'] = self.encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    df[f'{feature}_encoded'] = self.encoders[feature].transform(df[feature].astype(str))
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        
        numerical_features = [
            'amount_log', 'amount_sqrt', 'amount_user_zscore',
            'user_age', 'account_age_days', 'credit_score',
            'txn_count_last_1h', 'txn_count_last_24h',
            'amount_sum_last_1h', 'amount_sum_last_24h',
            'velocity_score', 'combined_risk_score'
        ]
        
        features_to_scale = [f for f in numerical_features if f in df.columns]
        
        if not hasattr(self, '_scaler_fitted'):
            self.scaler = StandardScaler()
            df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
            self._scaler_fitted = True
        else:
            df[features_to_scale] = self.scaler.transform(df[features_to_scale])
        
        return df