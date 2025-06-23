from pydantic import BaseModel, Field, validator
from typing import List, Optional
from decimal import Decimal

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    location: str = Field(..., description="Transaction location")
    device_type: str = Field(..., description="Device type (mobile, desktop, tablet)")
    is_foreign_transaction: bool = Field(False, description="Is transaction from foreign country")
    is_high_risk_country: bool = Field(False, description="Is transaction from high-risk country")
    card_type: Optional[str] = Field(None, description="Credit card type")
    payment_method: Optional[str] = Field(None, description="Payment method")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:  # 1M limit
            raise ValueError('Amount exceeds maximum limit')
        return v
    
    @validator('device_type')
    def validate_device_type(cls, v):
        allowed_devices = ['mobile', 'desktop', 'tablet', 'unknown']
        if v.lower() not in allowed_devices:
            raise ValueError(f'Device type must be one of: {allowed_devices}')
        return v.lower()

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_items=100)
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one transaction is required')
        return v