import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "../model/fraud_model.pkl")
model = joblib.load(model_path)

def get_model():
    return model
