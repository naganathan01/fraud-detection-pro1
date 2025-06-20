from pydantic import BaseModel

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    location: str
    device_type: str
    is_foreign_transaction: bool
    is_high_risk_country: bool
