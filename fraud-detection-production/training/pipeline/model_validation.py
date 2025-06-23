import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import joblib

class ModelValidator:
    """Model validation and performance assessment"""
    
    def __init__(self):
        self.validation_thresholds = {
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.85,
            'f1_score': 0.87,
            'auc': 0.95
        }
    
    def validate_models(self, test_data: pd.DataFrame) -> Dict:
        """Validate trained models against test data"""
        
        logging.info("Starting model validation...")
        
        # Load models
        models = self._load_models()
        
        # Prepare test data
        X_test, y_test = self._prepare_test_data(test_data)
        
        validation_results = {}
        
        for model_name, model in models.items():
            logging.info(f"Validating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Validate against thresholds
            validation_status = self._validate_against_thresholds(metrics)
            
            validation_results[model_name] = {
                'metrics': metrics,
                'validation_status': validation_status,
                'passed_validation': validation_status['overall_passed']
            }
            
            logging.info(f"{model_name} validation: {'✅ PASSED' if validation_status['overall_passed'] else '❌ FAILED'}")
        
        # Overall validation result
        overall_result = self._get_overall_validation_result(validation_results)
        
        logging.info(f"Overall validation: {'✅ PASSED' if overall_result['passed'] else '❌ FAILED'}")
        
        return overall_result
    
    def _load_models(self) -> Dict:
        """Load trained models"""
        models = {}
        
        model_files = {
            'xgboost': '/tmp/xgboost_model.pkl',
            'lightgbm': '/tmp/lightgbm_model.pkl',
            'random_forest': '/tmp/random_forest_model.pkl',
            'ensemble': '/tmp/ensemble_model.pkl'
        }
        
        for name, path in model_files.items():
            try:
                models[name] = joblib.load(path)
                logging.info(f"✅ Loaded {name} model")
            except Exception as e:
                logging.error(f"❌ Failed to load {name} model: {e}")
        
        return models
    
    def _prepare_test_data(self, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare test data for validation"""
        
        # Assume features are already engineered
        target_col = 'is_fraud'
        
        if target_col not in test_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Select feature columns (exclude target and metadata)
        feature_cols = [col for col in test_data.columns 
                       if col not in ['is_fraud', 'transaction_id', 'user_id', 'transaction_time']]
        
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        return X_test, y_test
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'support': len(y_true),
            'fraud_rate': y_true.mean()
        }
    
    def _validate_against_thresholds(self, metrics: Dict) -> Dict:
        """Validate metrics against thresholds"""
        
        validation_status = {
            'overall_passed': True,
            'individual_checks': {}
        }
        
        for metric_name, threshold in self.validation_thresholds.items():
            if metric_name in metrics:
                passed = metrics[metric_name] >= threshold
                validation_status['individual_checks'][metric_name] = {
                    'value': metrics[metric_name],
                    'threshold': threshold,
                    'passed': passed
                }
                
                if not passed:
                    validation_status['overall_passed'] = False
        
        return validation_status
    
    def _get_overall_validation_result(self, validation_results: Dict) -> Dict:
        """Get overall validation result across all models"""
        
        # Check if at least one model passed validation
        any_model_passed = any(
            result['passed_validation'] 
            for result in validation_results.values()
        )