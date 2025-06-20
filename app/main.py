from fastapi import FastAPI
from app.schemas import Transaction
from app.predict import predict_fraud

app = FastAPI()

@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_fraud(transaction.dict())
    return {"fraud": result}
