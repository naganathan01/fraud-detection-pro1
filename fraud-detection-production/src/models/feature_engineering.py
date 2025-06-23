# src/models/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, List, Tuple
import joblib
import logging

class ProductionFeatureEngine:
    """Production feature engineering for real-time inference"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ProductionFeatureEngine':
        """Fit feature engineering components on training data"""
        
        logging.info("Fitting feature engineering pipeline...")
        
        # Create features
        features_df = self._engineer_features(df)
        
        # Fit scalers and encoders
        self._fit_scalers(features_df)
        self._fit_encoders(features_df)
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        self.is_fitted = True
        
        logging.info(f"✅ Feature engineering fitted with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted components"""
        
        if not self.is_fitted:
            raise ValueError("Feature engineering pipeline not fitted yet")
        
        # Engineer features
        features_df = self._engineer_features(df)
        
        # Apply scaling and encoding
        features_df = self._apply_scaling(features_df)
        features_df = self._apply_encoding(features_df)
        
        # Ensure feature order consistency
        features_df = features_df.reindex(columns=self.feature_names, fill_value=0)
        
        return features_df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core feature engineering logic"""
        
        features_df = df.copy()
        
        # Amount features
        features_df['amount_log'] = np.log1p(features_df['amount'])
        features_df['amount_sqrt'] = np.sqrt(features_df['amount'])
        features_df['amount_rounded'] = (features_df['amount'] % 1 == 0).astype(int)
        
        # Time features (if transaction_time exists)
        if 'transaction_time' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['transaction_time']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df['transaction_time']).dt.dayofweek
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Categorical encoding
        if 'device_type' in features_df.columns:
            device_mapping = {'mobile': 0, 'desktop': 1, 'tablet': 2}
            features_df['device_encoded'] = features_df['device_type'].map(device_mapping).fillna(3)
        
        # Risk features
        if 'is_high_risk_country' in features_df.columns:
            features_df['country_risk'] = features_df['is_high_risk_country'].astype(int)
        
        return features_df
    
    def _fit_scalers(self, df: pd.DataFrame):
        """Fit numerical feature scalers"""
        
        numerical_features = [
            'amount_log', 'amount_sqrt', 'user_age', 'account_age_days', 'credit_score'
        ]
        
        for feature in numerical_features:
            if feature in df.columns:
                self.scalers[feature] = StandardScaler()
                self.scalers[feature].fit(df[[feature]])
    
    def _fit_encoders(self, df: pd.DataFrame):
        """Fit categorical feature encoders"""
        
        categorical_features = ['location', 'merchant_id']
        
        for feature in categorical_features:
            if feature in df.columns:
                self.encoders[feature] = LabelEncoder()
                self.encoders[feature].fit(df[feature].astype(str))
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers"""
        
        for feature, scaler in self.scalers.items():
            if feature in df.columns:
                df[feature] = scaler.transform(df[[feature]]).flatten()
        
        return df
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders"""
        
        for feature, encoder in self.encoders.items():
            if feature in df.columns:
                # Handle unseen categories
                try:
                    df[f'{feature}_encoded'] = encoder.transform(df[feature].astype(str))
                except ValueError:
                    # Use most frequent category for unseen values
                    most_frequent = encoder.classes_[0]
                    df[feature] = df[feature].fillna(most_frequent)
                    df[f'{feature}_encoded'] = encoder.transform(df[feature].astype(str))
        
        return df
    
    def save(self, filepath: str):
        """Save fitted feature engineering pipeline"""
        
        pipeline_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        logging.info(f"✅ Feature engineering pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ProductionFeatureEngine':
        """Load fitted feature engineering pipeline"""
        
        pipeline_data = joblib.load(filepath)
        
        instance = cls()
        instance.scalers = pipeline_data['scalers']
        instance.encoders = pipeline_data['encoders']
        instance.feature_names = pipeline_data['feature_names']
        instance.is_fitted = pipeline_data['is_fitted']
        
        logging.info(f"✅ Feature engineering pipeline loaded from {filepath}")
        
        return instance