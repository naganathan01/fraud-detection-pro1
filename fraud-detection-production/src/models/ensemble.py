# src/models/ensemble.py
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime

class FraudDetectionEnsemble:
    def __init__(self, model_paths: Dict[str, str], weights: Dict[str, float] = None):
        self.models = {}
        self.weights = weights or {'xgboost': 0.4, 'lightgbm': 0.4, 'neural_net': 0.2}
        self.feature_names = None
        self.threshold = 0.5
        
        # Load models
        for name, path in model_paths.items():
            try:
                self.models[name] = joblib.load(path)
                logging.info(f"✅ Loaded model: {name}")
            except Exception as e:
                logging.error(f"❌ Failed to load model {name}: {e}")
                
        self._validate_models()
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get fraud probability from ensemble"""
        if features.empty:
            return np.array([0.5])  # Default neutral probability
            
        predictions = []
        total_weight = 0
        
        for model_name, model in self.models.items():
            try:
                # Get prediction
                prob = model.predict_proba(features)[:, 1]  # Fraud probability
                weight = self.weights.get(model_name, 1.0)
                
                predictions.append(prob * weight)
                total_weight += weight
                
            except Exception as e:
                logging.warning(f"Model {model_name} prediction failed: {e}")
                continue
        
        if not predictions:
            return np.array([0.5])  # Fallback
            
        # Weighted ensemble
        ensemble_prob = np.sum(predictions, axis=0) / total_weight
        return ensemble_prob
    
    def predict(self, features: pd.DataFrame) -> Tuple[bool, float, Dict]:
        """Make fraud prediction with confidence and explanation"""
        try:
            # Get probability
            fraud_prob = self.predict_proba(features)[0]
            
            # Make binary decision
            is_fraud = fraud_prob > self.threshold
            
            # Generate explanation
            explanation = self._generate_explanation(features, fraud_prob)
            
            return is_fraud, fraud_prob, explanation
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            # Fallback to rule-based prediction
            return self._fallback_prediction(features)
    
    def _generate_explanation(self, features: pd.DataFrame, prob: float) -> Dict:
        """Generate prediction explanation"""
        return {
            'probability': float(prob),
            'confidence': 'high' if abs(prob - 0.5) > 0.3 else 'medium',
            'key_factors': self._get_key_factors(features),
            'model_version': 'ensemble_v1.0',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_key_factors(self, features: pd.DataFrame) -> List[str]:
        """Get key factors influencing prediction"""
        # Simplified - in production, use SHAP or LIME
        factors = []
        
        if features['amount'].iloc[0] > 1000:
            factors.append('high_amount')
        if features['txn_count_1h'].iloc[0] > 5:
            factors.append('high_velocity')
        if features.get('is_high_risk_country', [False]).iloc[0]:
            factors.append('high_risk_country')
            
        return factors
    
    def _fallback_prediction(self, features: pd.DataFrame) -> Tuple[bool, float, Dict]:
        """Rule-based fallback when models fail"""
        try:
            amount = features['amount'].iloc[0]
            is_high_risk = features.get('is_high_risk_country', [False]).iloc[0]
            txn_count = features.get('txn_count_1h', [0]).iloc[0]
            
            # Simple rule-based logic
            is_fraud = (amount > 5000 and is_high_risk) or txn_count > 10
            prob = 0.8 if is_fraud else 0.2
            
            explanation = {
                'probability': prob,
                'confidence': 'low',
                'key_factors': ['rule_based_fallback'],
                'model_version': 'fallback_v1.0',
                'timestamp': datetime.now().isoformat()
            }
            
            return is_fraud, prob, explanation
            
        except Exception as e:
            logging.error(f"Fallback prediction failed: {e}")
            return False, 0.5, {'error': 'prediction_failed'}
    
    def _validate_models(self):
        """Validate loaded models"""
        if not self.models:
            raise ValueError("No models loaded successfully")
            
        logging.info(f"✅ Ensemble initialized with {len(self.models)} models")