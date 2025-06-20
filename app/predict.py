from app.model import get_model
import pandas as pd

def predict_fraud(transaction: dict):
    # FORCE fraud=True for testing
    if transaction["amount"] > 9000 and transaction["is_high_risk_country"]:
        return True

    model = get_model()
    input_data = {k: transaction[k] for k in [
        "amount", "location", "device_type",
        "is_foreign_transaction", "is_high_risk_country"
    ]}
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return bool(prediction)

