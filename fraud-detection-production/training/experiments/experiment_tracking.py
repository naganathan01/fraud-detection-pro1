# training/experiments/experiment_tracking.py
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import joblib
import json

class ExperimentTracker:
    """MLflow experiment tracking for fraud detection models"""
    
    def __init__(self, experiment_name: str = "fraud_detection"):
        self.experiment_name = experiment_name
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            # Set tracking URI if configured
            # mlflow.set_tracking_uri("http://mlflow-server:5000")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logging.info(f"✅ Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logging.info(f"✅ Using existing experiment: {self.experiment_name}")
                
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logging.error(f"❌ Failed to setup MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Dict[str, str] = None):
        """Start MLflow run"""
        run_name = run_name or f"fraud_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name, tags=tags or {})
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log model hyperparameters"""
        try:
            # Convert complex objects to strings
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_params[key] = value
                else:
                    clean_params[key] = str(value)
            
            mlflow.log_params(clean_params)
            logging.info(f"✅ Logged {len(clean_params)} hyperparameters")
            
        except Exception as e:
            logging.error(f"❌ Failed to log hyperparameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log model performance metrics"""
        try:
            if step is not None:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value, step)
            else:
                mlflow.log_metrics(metrics)
            
            logging.info(f"✅ Logged {len(metrics)} metrics")
            
        except Exception as e:
            logging.error(f"❌ Failed to log metrics: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """Log trained model"""
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, model_name)
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, model_name)
            else:
                # Fallback to pickle
                mlflow.sklearn.log_model(model, model_name)
            
            logging.info(f"✅ Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logging.error(f"❌ Failed to log model {model_name}: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log training artifacts"""
        try:
            for artifact_name, file_path in artifacts.items():
                mlflow.log_artifact(file_path, artifact_name)
            
            logging.info(f"✅ Logged {len(artifacts)} artifacts")
            
        except Exception as e:
            logging.error(f"❌ Failed to log artifacts: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        try:
            # Log as parameters and metrics
            mlflow.log_param("dataset_size", dataset_info.get("size", 0))
            mlflow.log_param("dataset_features", dataset_info.get("n_features", 0))
            mlflow.log_metric("fraud_rate", dataset_info.get("fraud_rate", 0.0))
            
            # Save detailed info as artifact
            dataset_path = "/tmp/dataset_info.json"
            with open(dataset_path, 'w') as f:
                json.dump(dataset_info, f, indent=2, default=str)
            
            mlflow.log_artifact(dataset_path, "dataset_info")
            
            logging.info("✅ Logged dataset information")
            
        except Exception as e:
            logging.error(f"❌ Failed to log dataset info: {e}")
    
    def log_feature_importance(self, feature_names: list, importance_values: np.ndarray):
        """Log feature importance"""
        try:
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            # Save as CSV artifact
            importance_path = "/tmp/feature_importance.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path, "feature_importance")
            
            # Log top features as parameters
            top_features = feature_importance_df.head(10)
            for idx, row in top_features.iterrows():
                mlflow.log_param(f"top_feature_{idx+1}", row['feature'])
                mlflow.log_metric(f"top_feature_{idx+1}_importance", row['importance'])
            
            logging.info("✅ Logged feature importance")
            
        except Exception as e:
            logging.error(f"❌ Failed to log feature importance: {e}")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Log confusion matrix"""
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            cm_path = "/tmp/confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            
            # Log as artifact
            mlflow.log_artifact(cm_path, "confusion_matrix")
            
            # Log matrix values as metrics
            tn, fp, fn, tp = cm.ravel()
            mlflow.log_metric("true_negatives", tn)
            mlflow.log_metric("false_positives", fp)
            mlflow.log_metric("false_negatives", fn)
            mlflow.log_metric("true_positives", tp)
            
            logging.info("✅ Logged confusion matrix")
            
        except Exception as e:
            logging.error(f"❌ Failed to log confusion matrix: {e}")
    
    def register_model(self, model_name: str, model_uri: str, description: str = None):
        """Register model in MLflow Model Registry"""
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )
            
            logging.info(f"✅ Registered model {model_name} version {model_version.version}")
            return model_version
            
        except Exception as e:
            logging.error(f"❌ Failed to register model: {e}")
            raise
    
    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """Transition model to different stage"""
        try:
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logging.info(f"✅ Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logging.error(f"❌ Failed to transition model stage: {e}")
            raise
    
    def compare_models(self, run_ids: list) -> pd.DataFrame:
        """Compare multiple model runs"""
        try:
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            
            comparison_data = []
            for run_id in run_ids:
                run = client.get_run(run_id)
                
                comparison_data.append({
                    'run_id': run_id,
                    'run_name': run.info.run_name,
                    'accuracy': run.data.metrics.get('accuracy', None),
                    'precision': run.data.metrics.get('precision', None),
                    'recall': run.data.metrics.get('recall', None),
                    'f1_score': run.data.metrics.get('f1_score', None),
                    'model_type': run.data.params.get('model_type', None),
                    'start_time': run.info.start_time
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            logging.info(f"✅ Compared {len(run_ids)} model runs")
            
            return comparison_df
            
        except Exception as e:
            logging.error(f"❌ Failed to compare models: {e}")
            raise
    
    def get_best_model(self, metric: str = 'f1_score', order: str = 'DESC') -> Dict:
        """Get best performing model from experiment"""
        try:
            # Search for runs in current experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {order}"],
                max_results=1
            )
            
            if len(runs) > 0:
                best_run = runs.iloc[0]
                return {
                    'run_id': best_run['run_id'],
                    'run_name': best_run.get('tags.mlflow.runName', 'Unknown'),
                    'metrics': {
                        col.replace('metrics.', ''): best_run[col] 
                        for col in best_run.index 
                        if col.startswith('metrics.')
                    },
                    'params': {
                        col.replace('params.', ''): best_run[col] 
                        for col in best_run.index 
                        if col.startswith('params.')
                    }
                }
            else:
                return None
                
        except Exception as e:
            logging.error(f"❌ Failed to get best model: {e}")
            raise