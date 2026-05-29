# app/models.py
from pydantic import BaseModel, Field
from typing import Optional, List

# ─────────────────────────────────────
# INPUT MODELS
# ─────────────────────────────────────

class TransactionInput(BaseModel):
    """
    Raw transaction input from user/system
    These are the original features before
    any preprocessing
    """
    # Original features
    Time: float = Field(
        ...,
        description="Seconds elapsed since "
                    "first transaction",
        example=50000.0
    )
    Amount: float = Field(
        ...,
        description="Transaction amount in euros",
        example=150.0,
        ge=0  # must be >= 0
    )

    # PCA features V1-V28
    V1: float = Field(..., example=-1.3598)
    V2: float = Field(..., example=-0.0728)
    V3: float = Field(..., example=2.5364)
    V4: float = Field(..., example=1.3782)
    V5: float = Field(..., example=-0.3383)
    V6: float = Field(..., example=0.4624)
    V7: float = Field(..., example=0.2396)
    V8: float = Field(..., example=0.0987)
    V9: float = Field(..., example=0.3638)
    V10: float = Field(..., example=0.0908)
    V11: float = Field(..., example=-0.5516)
    V12: float = Field(..., example=-0.6178)
    V13: float = Field(..., example=-0.9914)
    V14: float = Field(..., example=-0.3112)
    V15: float = Field(..., example=1.4681)
    V16: float = Field(..., example=-0.4704)
    V17: float = Field(..., example=0.2076)
    V18: float = Field(..., example=0.0258)
    V19: float = Field(..., example=0.4039)
    V20: float = Field(..., example=0.2514)
    V21: float = Field(..., example=-0.0183)
    V22: float = Field(..., example=0.2778)
    V23: float = Field(..., example=-0.1105)
    V24: float = Field(..., example=0.0669)
    V25: float = Field(..., example=0.1285)
    V26: float = Field(..., example=-0.1891)
    V27: float = Field(..., example=0.1336)
    V28: float = Field(..., example=-0.0210)

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 50000.0,
                "Amount": 150.0,
                "V1": -1.3598,
                "V2": -0.0728,
                "V3": 2.5364,
                "V4": 1.3782,
                "V5": -0.3383,
                "V6": 0.4624,
                "V7": 0.2396,
                "V8": 0.0987,
                "V9": 0.3638,
                "V10": 0.0908,
                "V11": -0.5516,
                "V12": -0.6178,
                "V13": -0.9914,
                "V14": -0.3112,
                "V15": 1.4681,
                "V16": -0.4704,
                "V17": 0.2076,
                "V18": 0.0258,
                "V19": 0.4039,
                "V20": 0.2514,
                "V21": -0.0183,
                "V22": 0.2778,
                "V23": -0.1105,
                "V24": 0.0669,
                "V25": 0.1285,
                "V26": -0.1891,
                "V27": 0.1336,
                "V28": -0.0210
            }
        }


# ─────────────────────────────────────
# OUTPUT MODELS
# ─────────────────────────────────────

class FeatureContribution(BaseModel):
    """Single feature SHAP contribution"""
    feature: str
    value: float
    impact: float


class SHAPExplanation(BaseModel):
    """SHAP explanation for prediction"""
    base_rate: float
    top_fraud_signals: List[FeatureContribution]
    top_legit_signals: List[FeatureContribution]


class PredictionResponse(BaseModel):
    """Complete fraud prediction response"""
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold_used: float
    explanation: Optional[SHAPExplanation]
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    auc_roc: float
    threshold: float


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    version: str
    auc_roc: float
    fraud_precision: float
    fraud_recall: float
    fraud_f1: float
    threshold: float
    training_data: str
    total_features: int