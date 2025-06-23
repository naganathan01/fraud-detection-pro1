# src/models/model_loader.py
import joblib
import mlflow
import boto3
import os
import logging
from typing import Dict, Optional, Any
from src.utils.config import settings

class ModelLoader:
    """Load and manage ML models from various sources"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.s3_client = boto3.client('s3') if settings.AWS_ACCESS_KEY_ID else None
        
    def load_model_from_file(self, model_name: str, filepath: str) -> Any:
        """Load model from local file"""
        
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'source': 'file',
                'path': filepath,
                'loaded_at': datetime.now().isoformat()
            }
            
            logging.info(f"✅ Loaded {model_name} from {filepath}")
            return model
            
        except Exception as e:
            logging.error(f"❌ Failed to load {model_name} from {filepath}: {e}")
            raise
    
    def load_model_from_mlflow(self, model_name: str, model_uri: str) -> Any:
        """Load model from MLflow registry"""
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'source': 'mlflow',
                'uri': model_uri,
                'loaded_at': datetime.now().isoformat()
            }
            
            logging.info(f"✅ Loaded {model_name} from MLflow: {model_uri}")
            return model
            
        except Exception as e:
            logging.error(f"❌ Failed to load {model_name} from MLflow: {e}")
            raise
    
    def load_model_from_s3(self, model_name: str, bucket: str, key: str) -> Any:
        """Load model from S3"""
        
        if not self.s3_client:
            raise ValueError("S3 client not configured")
        
        try:
            # Download model file
            local_path = f"/tmp/{model_name}.pkl"
            self.s3_client.download_file(bucket, key, local_path)
            
            # Load model
            model = joblib.load(local_path)
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'source': 's3',
                'bucket': bucket,
                'key': key,
                'loaded_at': datetime.now().isoformat()
            }
            
            # Cleanup
            os.remove(local_path)
            
            logging.info(f"✅ Loaded {model_name} from S3: s3://{bucket}/{key}")
            return model
            
        except Exception as e:
            logging.error(f"❌ Failed to load {model_name} from S3: {e}")
            raise
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model by name"""
        return self.models.get(model_name)
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get model metadata"""
        return self.model_metadata.get(model_name)
    
    def list_models(self) -> Dict[str, Dict]:
        """List all loaded models with metadata"""
        return {
            name: {
                'metadata': self.model_metadata.get(name, {}),
                'loaded': True
            }
            for name in self.models.keys()
        }
    
    def reload_model(self, model_name: str) -> Any:
        """Reload a specific model"""
        
        if model_name not in self.model_metadata:
            raise ValueError(f"Model {model_name} not found")
        
        metadata = self.model_metadata[model_name]
        
        if metadata['source'] == 'file':
            return self.load_model_from_file(model_name, metadata['path'])
        elif metadata['source'] == 'mlflow':
            return self.load_model_from_mlflow(model_name, metadata['uri'])
        elif metadata['source'] == 's3':
            return self.load_model_from_s3(model_name, metadata['bucket'], metadata['key'])
        else:
            raise ValueError(f"Unknown source: {metadata['source']}")
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        
        if model_name in self.models:
            del self.models[model_name]
            logging.info(f"✅ Unloaded model: {model_name}")
        
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]