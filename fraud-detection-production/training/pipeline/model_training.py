# training/pipeline/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import logging
from typing import Dict, Tuple

class ModelTrainer:
    """Production model training with hyperparameter optimization"""
    
    def __init__(self):
        self.models = {
            'xgboost': XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'neural_net': MLPClassifier(random_state=42)
        }
        
        self.param_grids = {
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'neural_net': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
    
    def train_ensemble(self, data: pd.DataFrame) -> Dict:
        """Train ensemble of models with hyperparameter optimization"""
        
        # Prepare data
        X, y = self._prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trained_models = {}
        training_metrics = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logging.info(f"Training {model_name}...")
            
            # Hyperparameter optimization
            best_model = self._optimize_hyperparameters(
                model, self.param_grids[model_name], X_train, y_train
            )
            
            # Train best model
            best_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Store results
            trained_models[model_name] = best_model
            training_metrics[model_name] = metrics
            
            # Log to MLflow
            mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items()})
            
            logging.info(f"✅ {model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Create ensemble
        ensemble_model = self._create_ensemble(trained_models, X_test, y_test)
        
        # Save models
        model_artifacts = self._save_models(trained_models, ensemble_model)
        
        return {
            'models': trained_models,
            'ensemble': ensemble_model,
            'metrics': training_metrics,
            'artifacts': model_artifacts
        }
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Assume 'is_fraud' is the target column
        target_col = 'is_fraud'
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Handle missing values
        X = X.fillna(X.median() for col in X.select_dtypes(include=[np.number]).columns)
        X = X.fillna(X.mode().iloc[0] for col in X.select_dtypes(include=['object']).columns)
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        return X, y
    
    def _optimize_hyperparameters(self, model, param_grid: Dict, X_train: pd.DataFrame, y_train: pd.Series):
        """Optimize hyperparameters using GridSearchCV"""
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        return grid_search.best_estimator_
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
    
    def _create_ensemble(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series):
        """Create weighted ensemble based on performance"""
        
        # Calculate weights based on F1 scores
        weights = {}
        total_score = 0
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            weights[model_name] = f1
            total_score += f1
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_score
        
        logging.info(f"Ensemble weights: {weights}")
        
        return {
            'models': models,
            'weights': weights
        }
    
    def _save_models(self, models: Dict, ensemble: Dict) -> Dict:
        """Save trained models"""
        artifacts = {}
        
        # Save individual models
        for model_name, model in models.items():
            model_path = f"/tmp/{model_name}_model.pkl"
            joblib.dump(model, model_path)
            artifacts[f"{model_name}_model"] = model_path
        
        # Save ensemble
        ensemble_path = "/tmp/ensemble_model.pkl"
        joblib.dump(ensemble, ensemble_path)
        artifacts['ensemble_model'] = ensemble_path
        
        return artifacts