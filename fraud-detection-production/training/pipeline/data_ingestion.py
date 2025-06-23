import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple
import boto3
from sqlalchemy import create_engine
from src.utils.config import settings

class DataIngestion:
    """Data ingestion for model training"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.db_engine = create_engine(settings.DATABASE_URL)
        
    def extract_training_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract training data from data warehouse"""
        
        query = """
        SELECT 
            t.transaction_id,
            t.user_id,
            t.amount,
            t.merchant_id,
            t.location,
            t.device_type,
            t.is_foreign_transaction,
            t.is_high_risk_country,
            t.transaction_time,
            CASE 
                WHEN f.fraud_flag IS NOT NULL THEN f.fraud_flag
                ELSE 0 
            END as is_fraud,
            -- User features
            u.user_age,
            u.account_age_days,
            u.credit_score,
            -- Derived features
            EXTRACT(HOUR FROM t.transaction_time) as hour_of_day,
            EXTRACT(DOW FROM t.transaction_time) as day_of_week,
            DATE_PART('epoch', t.transaction_time - LAG(t.transaction_time) 
                     OVER (PARTITION BY t.user_id ORDER BY t.transaction_time)) / 3600 as hours_since_last_txn
        FROM transactions t
        LEFT JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        LEFT JOIN users u ON t.user_id = u.user_id
        WHERE t.transaction_time BETWEEN %s AND %s
        AND t.amount > 0
        ORDER BY t.transaction_time
        """
        
        logging.info(f"Extracting training data from {start_date} to {end_date}")
        
        try:
            df = pd.read_sql_query(
                query, 
                self.db_engine, 
                params=[start_date, end_date]
            )
            
            # Data quality checks
            df = self._clean_data(df)
            
            logging.info(f"✅ Extracted {len(df)} training samples")
            logging.info(f"Fraud rate: {df['is_fraud'].mean():.3f}")
            
            return df
            
        except Exception as e:
            logging.error(f"Data extraction failed: {e}")
            # Fallback to synthetic data for demo
            return self._generate_synthetic_data(10000)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['transaction_id'])
        
        # Handle missing values
        df['hours_since_last_txn'] = df['hours_since_last_txn'].fillna(24.0)
        df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
        
        # Remove outliers (amounts > 99.9th percentile)
        amount_threshold = df['amount'].quantile(0.999)
        df = df[df['amount'] <= amount_threshold]
        
        # Ensure fraud labels are binary
        df['is_fraud'] = df['is_fraud'].astype(int)
        
        logging.info(f"Data cleaning completed. Final shape: {df.shape}")
        
        return df
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic training data for demo purposes"""
        
        np.random.seed(42)
        
        # Generate base features
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
            'user_id': [f'USER_{i:06d}' for i in np.random.randint(0, n_samples//10, n_samples)],
            'amount': np.random.lognormal(mean=4, sigma=1.5, size=n_samples),
            'merchant_id': [f'MERCHANT_{i:04d}' for i in np.random.randint(0, 1000, n_samples)],
            'location': np.random.choice(['US', 'CA', 'UK', 'DE', 'FR'], n_samples),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.6, 0.3, 0.1]),
            'is_foreign_transaction': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
            'is_high_risk_country': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'user_age': np.random.normal(35, 12, n_samples).clip(18, 80),
            'account_age_days': np.random.exponential(365, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
            'hours_since_last_txn': np.random.exponential(24, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate fraud labels based on rules
        fraud_prob = (
            (df['amount'] > 1000).astype(int) * 0.3 +
            df['is_high_risk_country'].astype(int) * 0.4 +
            (df['hour_of_day'].isin([0, 1, 2, 3])).astype(int) * 0.2 +
            (df['hours_since_last_txn'] < 0.1).astype(int) * 0.5
        )
        
        df['is_fraud'] = np.random.binomial(1, fraud_prob.clip(0, 0.8))
        
        logging.info(f"✅ Generated {len(df)} synthetic training samples")
        logging.info(f"Synthetic fraud rate: {df['is_fraud'].mean():.3f}")
        
        return df