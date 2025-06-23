# src/monitoring/drift_detection.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

class DataDriftDetector:
    """Real-time data drift detection using statistical methods"""
    
    def __init__(self, reference_data: pd.DataFrame = None):
        self.reference_data = reference_data
        self.drift_threshold = 0.1  # PSI threshold
        self.feature_stats = {}
        
        if reference_data is not None:
            self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute reference statistics for drift detection"""
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['float64', 'int64']:
                self.feature_stats[column] = {
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'percentiles': np.percentile(self.reference_data[column], 
                                               [10, 25, 50, 75, 90])
                }
            else:
                # Categorical features
                self.feature_stats[column] = {
                    'value_counts': self.reference_data[column].value_counts(normalize=True)
                }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift in current data vs reference"""
        drift_results = {
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'timestamp': datetime.now().isoformat()
        }
        
        total_psi = 0
        features_with_drift = 0
        
        for feature in current_data.columns:
            if feature in self.feature_stats:
                psi_score = self._calculate_psi(feature, current_data[feature])
                
                drift_results['feature_drift'][feature] = {
                    'psi_score': psi_score,
                    'drift_detected': psi_score > self.drift_threshold,
                    'severity': self._get_drift_severity(psi_score)
                }
                
                total_psi += psi_score
                if psi_score > self.drift_threshold:
                    features_with_drift += 1
        
        # Overall drift assessment
        if len(self.feature_stats) > 0:
            avg_psi = total_psi / len(self.feature_stats)
            drift_results['drift_score'] = avg_psi
            drift_results['overall_drift_detected'] = (
                avg_psi > self.drift_threshold or 
                features_with_drift > len(self.feature_stats) * 0.3
            )
        
        # Log drift if detected
        if drift_results['overall_drift_detected']:
            logging.warning(f"🚨 Data drift detected! Score: {drift_results['drift_score']:.3f}")
            self._trigger_drift_alert(drift_results)
        
        return drift_results
    
    def _calculate_psi(self, feature: str, current_values: pd.Series) -> float:
        """Calculate Population Stability Index"""
        try:
            if feature not in self.feature_stats:
                return 0.0
                
            if current_values.dtype in ['float64', 'int64']:
                # Numerical feature PSI
                return self._calculate_numerical_psi(feature, current_values)
            else:
                # Categorical feature PSI
                return self._calculate_categorical_psi(feature, current_values)
                
        except Exception as e:
            logging.error(f"PSI calculation error for {feature}: {e}")
            return 0.0
    
    def _calculate_numerical_psi(self, feature: str, current_values: pd.Series) -> float:
        """Calculate PSI for numerical features"""
        ref_percentiles = self.feature_stats[feature]['percentiles']
        
        # Create bins based on reference percentiles
        bins = [-np.inf] + list(ref_percentiles) + [np.inf]
        
        # Get reference distribution (uniform since we use percentiles)
        ref_counts = np.ones(len(bins) - 1) / (len(bins) - 1)
        
        # Get current distribution
        current_counts, _ = np.histogram(current_values, bins=bins, density=True)
        current_counts = current_counts / current_counts.sum()
        
        # Avoid zero values
        ref_counts = np.maximum(ref_counts, 1e-8)
        current_counts = np.maximum(current_counts, 1e-8)
        
        # Calculate PSI
        psi = np.sum((current_counts - ref_counts) * np.log(current_counts / ref_counts))
        return psi
    
    def _calculate_categorical_psi(self, feature: str, current_values: pd.Series) -> float:
        """Calculate PSI for categorical features"""
        ref_dist = self.feature_stats[feature]['value_counts']
        current_dist = current_values.value_counts(normalize=True)
        
        # Align distributions
        all_categories = set(ref_dist.index) | set(current_dist.index)
        
        ref_aligned = []
        current_aligned = []
        
        for category in all_categories:
            ref_aligned.append(ref_dist.get(category, 1e-8))
            current_aligned.append(current_dist.get(category, 1e-8))
        
        ref_aligned = np.array(ref_aligned)
        current_aligned = np.array(current_aligned)
        
        # Calculate PSI
        psi = np.sum((current_aligned - ref_aligned) * np.log(current_aligned / ref_aligned))
        return psi
    
    def _get_drift_severity(self, psi_score: float) -> str:
        """Categorize drift severity"""
        if psi_score < 0.1:
            return 'no_drift'
        elif psi_score < 0.2:
            return 'moderate_drift'
        else:
            return 'severe_drift'
    
    def _trigger_drift_alert(self, drift_results: Dict):
        """Trigger alert when drift is detected"""
        # In production, integrate with alerting system (PagerDuty, Slack, etc.)
        alert_message = f"""
        🚨 DATA DRIFT ALERT 🚨
        
        Drift Score: {drift_results['drift_score']:.3f}
        Features Affected: {len([f for f, d in drift_results['feature_drift'].items() if d['drift_detected']])}
        Timestamp: {drift_results['timestamp']}
        
        Action Required: Review model performance and consider retraining
        """
        
        logging.critical(alert_message)
        # TODO: Send to alerting system