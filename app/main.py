# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import time

from app.models import (
    TransactionInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from app.utils import load_components, predict

# ─────────────────────────────────────
# INITIALIZE APP
# ─────────────────────────────────────

app = FastAPI(
    title="🔍 Fraud Detection API",
    description="""
    Real-time fraud detection using XGBoost + SHAP

    ## Features
    * **Predict** — Get fraud probability for any transaction
    * **Explain** — SHAP explanation for every prediction
    * **Health** — Check API and model status
    * **Info** — Get model performance metrics

    ## Model
    * Algorithm: XGBoost
    * AUC-ROC: 0.9817
    * Threshold: 0.5 (high recall)
    * Training: Original data + scale_pos_weight
    """,
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc UI
)

# Allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load components at startup
print("Loading model components...")
components = load_components()
print("✅ All components loaded!")


# ─────────────────────────────────────
# ROUTES
# ─────────────────────────────────────

@app.get("/",
         summary="Welcome",
         tags=["General"])
def root():
    """Welcome message"""
    return {
        "message": "🔍 Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health",
         response_model=HealthResponse,
         summary="Health Check",
         tags=["General"])
def health_check():
    """
    Check if API and model are running correctly
    """
    config = components['config']
    return {
        "status": "healthy ✅",
        "model_loaded": True,
        "model_version": "1.0.0",
        "auc_roc": config.get('auc_roc', 0.9817),
        "threshold": config.get('threshold', 0.5)
    }


@app.get("/model-info",
         response_model=ModelInfoResponse,
         summary="Model Information",
         tags=["General"])
def model_info():
    """
    Get detailed model performance information
    """
    config = components['config']
    model = components['model']

    return {
        "model_type": "XGBoost",
        "version": "1.0.0",
        "auc_roc": config.get('auc_roc', 0.9817),
        "fraud_precision": config.get(
            'fraud_precision', 0.8542
        ),
        "fraud_recall": config.get(
            'fraud_recall', 0.8367
        ),
        "fraud_f1": config.get('fraud_f1', 0.8454),
        "threshold": config.get('threshold', 0.5),
        "training_data": "Original (no SMOTE)",
        "total_features": model.n_features_in_
    }


@app.post("/predict",
          response_model=PredictionResponse,
          summary="Predict Fraud",
          tags=["Prediction"])
def predict_fraud(transaction: TransactionInput):
    """
    Predict if a transaction is fraudulent.

    Returns:
    - **fraud_probability**: 0 to 1
    - **is_fraud**: True/False
    - **risk_level**: LOW/MEDIUM/HIGH
    - **explanation**: SHAP feature contributions
    """
    try:
        # Convert to dict
        transaction_dict = transaction.dict()

        # Get prediction
        result = predict(
            transaction_dict,
            components,
            return_explanation=True
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/fast",
          summary="Fast Predict (No SHAP)",
          tags=["Prediction"])
def predict_fast(transaction: TransactionInput):
    """
    Fast prediction without SHAP explanation.
    Use when speed is critical (< 10ms response)
    """
    try:
        transaction_dict = transaction.dict()

        result = predict(
            transaction_dict,
            components,
            return_explanation=False  # skip SHAP
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch",
          summary="Batch Predict",
          tags=["Prediction"])
def predict_batch(
    transactions: list[TransactionInput]
):
    """
    Predict multiple transactions at once.
    Maximum 100 transactions per request.
    """
    if len(transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 transactions per batch"
        )

    results = []
    for transaction in transactions:
        try:
            result = predict(
                transaction.dict(),
                components,
                return_explanation=False
            )
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "is_fraud": None
            })

    return {
        "total": len(transactions),
        "fraud_count": sum(
            1 for r in results
            if r.get('is_fraud') == True
        ),
        "results": results
    }