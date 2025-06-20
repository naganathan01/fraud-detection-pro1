import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

# Simulated training data
X = pd.DataFrame({
    'amount': np.random.rand(100),
    'location': np.random.randint(0, 5, 100),
    'device_type': np.random.randint(0, 3, 100),
    'is_foreign_transaction': np.random.randint(0, 2, 100),
    'is_high_risk_country': np.random.randint(0, 2, 100)
})
y = np.random.randint(0, 2, 100)

# Train and save the model
model = RandomForestClassifier()
model.fit(X, y)

# Make sure the model folder exists
os.makedirs("model", exist_ok=True)

# Save the model
joblib.dump(model, "model/fraud_model.pkl")
print("✅ Model saved successfully at: model/fraud_model.pkl")
