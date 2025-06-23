# infrastructure/airflow/dags/data_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import pandas as pd
import logging

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'fraud_data_pipeline',
    default_args=default_args,
    description='Daily data processing pipeline for fraud detection',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
)

def extract_daily_data(**context):
    """Extract daily transaction data"""
    import logging
    from training.pipeline.data_ingestion import DataIngestion
    
    # Get execution date
    execution_date = context['execution_date']
    start_date = execution_date
    end_date = execution_date + timedelta(days=1)
    
    logging.info(f"Extracting data for {start_date} to {end_date}")
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Extract data
    daily_data = data_ingestion.extract_training_data(start_date, end_date)
    
    # Save to staging
    daily_data.to_parquet(f'/tmp/daily_data_{execution_date.strftime("%Y%m%d")}.parquet', index=False)
    
    logging.info(f"✅ Extracted {len(daily_data)} daily transactions")
    return len(daily_data)

def validate_daily_data(**context):
    """Validate daily data quality"""
    execution_date = context['execution_date']
    
    # Load daily data
    daily_data = pd.read_parquet(f'/tmp/daily_data_{execution_date.strftime("%Y%m%d")}.parquet')
    
    # Data quality checks
    checks = {
        'row_count': len(daily_data) > 0,
        'no_nulls_in_key_fields': daily_data[['transaction_id', 'user_id', 'amount']].isnull().sum().sum() == 0,
        'amount_positive': (daily_data['amount'] > 0).all(),
        'fraud_rate_reasonable': 0 <= daily_data['is_fraud'].mean() <= 0.2
    }
    
    if not all(checks.values()):
        failed_checks = [k for k, v in checks.items() if not v]
        raise ValueError(f"Data quality checks failed: {failed_checks}")
    
    logging.info("✅ Daily data quality validation passed")
    return checks

def process_features(**context):
    """Process features for daily data"""
    execution_date = context['execution_date']
    
    # Load data
    daily_data = pd.read_parquet(f'/tmp/daily_data_{execution_date.strftime("%Y%m%d")}.parquet')
    
    # Feature engineering (simplified)
    from src.features.batch_features import BatchFeatureEngine
    
    feature_engine = BatchFeatureEngine()
    processed_data = feature_engine.process_batch_features(daily_data)
    
    # Save processed data
    processed_data.to_parquet(f'/tmp/processed_data_{execution_date.strftime("%Y%m%d")}.parquet', index=False)
    
    logging.info(f"✅ Processed features for {len(processed_data)} transactions")
    return processed_data.shape

def update_feature_store(**context):
    """Update feature store with daily aggregations"""
    execution_date = context['execution_date']
    
    # Load processed data
    processed_data = pd.read_parquet(f'/tmp/processed_data_{execution_date.strftime("%Y%m%d")}.parquet')
    
    # Update feature store (simplified)
    from src.features.feature_store import FeatureStore
    
    feature_store = FeatureStore()
    
    # Update user aggregations
    for user_id in processed_data['user_id'].unique():
        user_data = processed_data[processed_data['user_id'] == user_id]
        
        # Calculate daily aggregations
        daily_stats = {
            'txn_count_daily': len(user_data),
            'amount_sum_daily': user_data['amount'].sum(),
            'avg_amount_daily': user_data['amount'].mean(),
            'fraud_count_daily': user_data['is_fraud'].sum()
        }
        
        # Update feature store (in production, this would update Redis)
        # For now, just log the update
        logging.info(f"Updated features for user {user_id}: {daily_stats}")
    
    logging.info("✅ Feature store updated with daily aggregations")
    return len(processed_data['user_id'].unique())

# Define tasks
extract_task = PythonOperator(
    task_id='extract_daily_data',
    python_callable=extract_daily_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_daily_data',
    python_callable=validate_daily_data,
    dag=dag
)

process_task = PythonOperator(
    task_id='process_features',
    python_callable=process_features,
    dag=dag
)

update_task = PythonOperator(
    task_id='update_feature_store',
    python_callable=update_feature_store,
    dag=dag
)

# Define dependencies
extract_task >> validate_task >> process_task >> update_task