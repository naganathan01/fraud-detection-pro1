# infrastructure/airflow/dags/model_training_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
import pandas as pd
import numpy as np

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'fraud_model_training',
    default_args=default_args,
    description='Automated fraud detection model training pipeline',
    schedule_interval='@weekly',  # Weekly retraining
    catchup=False,
    max_active_runs=1
)

def extract_training_data(**context):
    """Extract training data from data warehouse"""
    import logging
    from training.pipeline.data_ingestion import DataIngestion
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Extract data for last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    logging.info(f"Extracting training data from {start_date} to {end_date}")
    
    # Extract features and labels
    training_data = data_ingestion.extract_training_data(start_date, end_date)
    
    # Save to staging area
    training_data.to_parquet('/tmp/training_data.parquet', index=False)
    
    logging.info(f"✅ Extracted {len(training_data)} training samples")
    return len(training_data)

def validate_data_quality(**context):
    """Validate data quality before training"""
    import logging
    from training.pipeline.data_validation import DataValidator
    
    # Load training data
    training_data = pd.read_parquet('/tmp/training_data.parquet')
    
    # Initialize validator
    validator = DataValidator()
    
    # Run validation checks
    validation_results = validator.validate(training_data)
    
    if not validation_results['is_valid']:
        raise ValueError(f"Data validation failed: {validation_results['errors']}")
    
    logging.info("✅ Data quality validation passed")
    return validation_results

def engineer_features(**context):
    """Engineer features for model training"""
    import logging
    from training.pipeline.feature_engineering import FeatureEngineer
    
    # Load raw data
    raw_data = pd.read_parquet('/tmp/training_data.parquet')
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Engineer features
    feature_data = feature_engineer.engineer_features(raw_data)
    
    # Save engineered features
    feature_data.to_parquet('/tmp/feature_data.parquet', index=False)
    
    logging.info(f"✅ Engineered {feature_data.shape[1]} features")
    return feature_data.shape

def train_models(**context):
    """Train ensemble models"""
    import logging
    import mlflow
    from training.pipeline.model_training import ModelTrainer
    
    # Load feature data
    feature_data = pd.read_parquet('/tmp/feature_data.parquet')
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"fraud_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Train models
        training_results = trainer.train_ensemble(feature_data)
        
        # Log metrics
        mlflow.log_metrics(training_results['metrics'])
        mlflow.log_artifacts(training_results['artifacts'])
        
        # Save models
        for model_name, model in training_results['models'].items():
            mlflow.sklearn.log_model(model, model_name)
        
        logging.info("✅ Model training completed")
        return training_results['metrics']

def validate_models(**context):
    """Validate trained models"""
    import logging
    from training.pipeline.model_validation import ModelValidator
    
    # Initialize validator
    validator = ModelValidator()
    
    # Load test data
    test_data = pd.read_parquet('/tmp/test_data.parquet')
    
    # Validate models
    validation_results = validator.validate_models(test_data)
    
    # Check if models meet quality thresholds
    if validation_results['accuracy'] < 0.92:
        raise ValueError(f"Model accuracy {validation_results['accuracy']:.3f} below threshold 0.92")
    
    if validation_results['precision'] < 0.90:
        raise ValueError(f"Model precision {validation_results['precision']:.3f} below threshold 0.90")
    
    logging.info("✅ Model validation passed")
    return validation_results

def deploy_models(**context):
    """Deploy validated models to production"""
    import logging
    from training.deployment.model_deployment import ModelDeployer
    
    # Initialize deployer
    deployer = ModelDeployer()
    
    # Deploy to staging first
    staging_deployment = deployer.deploy_to_staging()
    
    # Run smoke tests
    smoke_test_results = deployer.run_smoke_tests()
    
    if smoke_test_results['success']:
        # Deploy to production with canary strategy
        production_deployment = deployer.deploy_to_production(strategy='canary')
        logging.info("✅ Models deployed to production")
        return production_deployment
    else:
        raise ValueError(f"Smoke tests failed: {smoke_test_results['errors']}")

# Define tasks
extract_data_task = PythonOperator(
    task_id='extract_training_data',
    python_callable=extract_training_data,
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_data_quality', 
    python_callable=validate_data_quality,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

validate_models_task = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models,
    dag=dag
)

deploy_models_task = PythonOperator(
    task_id='deploy_models',
    python_callable=deploy_models,
    dag=dag
)

# Define dependencies
extract_data_task >> validate_data_task >> feature_engineering_task
feature_engineering_task >> train_models_task >> validate_models_task
validate_models_task >> deploy_models_task