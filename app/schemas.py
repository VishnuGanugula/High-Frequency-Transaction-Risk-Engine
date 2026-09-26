from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TransactionInput(BaseModel):
    step: int = Field(..., description="Hour of transaction simulation step", json_schema_extra={"example": 1})
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN", json_schema_extra={"example": "TRANSFER"})
    amount: float = Field(..., description="Amount of transaction", json_schema_extra={"example": 181.0})
    nameOrig: str = Field(..., description="Customer origin account ID", json_schema_extra={"example": "C1305486145"})
    oldbalanceOrg: float = Field(..., description="Initial balance before transaction", json_schema_extra={"example": 181.0})
    newbalanceOrig: float = Field(..., description="New balance after transaction", json_schema_extra={"example": 0.0})
    nameDest: str = Field(..., description="Recipient account ID", json_schema_extra={"example": "C553264065"})
    oldbalanceDest: float = Field(..., description="Initial destination balance before transaction", json_schema_extra={"example": 0.0})
    newbalanceDest: float = Field(..., description="New destination balance after transaction", json_schema_extra={"example": 0.0})

class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

class PredictionResult(BaseModel):
    fraud_detected: bool = Field(..., description="Cost-sensitive fraud decision based on threshold 0.09")
    probability: float = Field(..., description="Calibrated fraud probability")
    cost_sensitive_threshold: float = 0.09
    risk_level: str = Field(..., description="Risk tier: CRITICAL, HIGH, MEDIUM, LOW")
    decision: str = Field(..., description="ACTION: REJECT, REVIEW, or APPROVE")
    nameOrig: str
    amount: float
    timestamp: str

class ExplanationResult(BaseModel):
    fraud_detected: bool
    probability: float
    risk_level: str
    top_risk_factors: Dict[str, float]
    shap_summary: Dict[str, Any]

class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
    version: str
