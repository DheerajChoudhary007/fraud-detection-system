# app/utils.py
import numpy as np
import pandas as pd
import pickle
import json
import os

# ─────────────────────────────────────
# LOAD ALL COMPONENTS
# ─────────────────────────────────────

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_components():
    """Load all model components"""
    components = {}

    # Load model
    model_path = os.path.join(
        MODELS_DIR, 'xgb_fraud_model.pkl'
    )
    with open(model_path, 'rb') as f:
        components['model'] = pickle.load(f)

    # Load SHAP explainer
    shap_path = os.path.join(
        MODELS_DIR, 'shap_explainer.pkl'
    )
    with open(shap_path, 'rb') as f:
        components['explainer'] = pickle.load(f)

    # Load scalers
    scaler_amount_path = os.path.join(
        MODELS_DIR, 'scaler_amount.pkl'
    )
    with open(scaler_amount_path, 'rb') as f:
        components['scaler_amount'] = pickle.load(f)

    scaler_hour_path = os.path.join(
        MODELS_DIR, 'scaler_hour.pkl'
    )
    with open(scaler_hour_path, 'rb') as f:
        components['scaler_hour'] = pickle.load(f)

    # Load threshold config
    config_path = os.path.join(
        MODELS_DIR, 'threshold_config.json'
    )
    with open(config_path, 'r') as f:
        components['config'] = json.load(f)

    return components


# ─────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────

def preprocess(transaction_dict, components):
    """
    Preprocess raw transaction into
    model-ready format

    Steps:
    1. Create hour from Time
    2. Log transform Amount
    3. Scale Amount and hour
    4. Drop Time and Amount
    5. Return correct column order
    """
    df = pd.DataFrame([transaction_dict])

    # Feature engineering (same as Day 4)
    df['hour'] = (df['Time'] // 3600) % 24
    df['amount_log'] = np.log(df['Amount'] + 1)
    df['amount_scaled'] = \
        components['scaler_amount'].transform(
            df[['Amount']]
        )
    df['hour_scaled'] = \
        components['scaler_hour'].transform(
            df[['hour']]
        )

    # Drop original columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    return df


# ─────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────

def predict(transaction_dict,
            components,
            return_explanation=True):
    """
    Full prediction pipeline:
    preprocess → predict → explain
    """

    # Preprocess
    processed = preprocess(
        transaction_dict, components
    )

    # Get probability
    prob = float(
        components['model'].predict_proba(
            processed
        )[0][1]
    )

    # Apply threshold
    final_threshold = components['config']['final_threshold']
    is_fraud = prob >= final_threshold

    # Risk level
    if prob >= 0.7:
        risk = "HIGH"
    elif prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    result = {
        'fraud_probability': round(prob, 4),
        'is_fraud': bool(is_fraud),
        'risk_level': risk,
        'threshold_used': final_threshold,
        'model_version': '1.0.0'
    }

    # SHAP explanation
    if return_explanation:
        shap_vals = components['explainer']\
            .shap_values(processed)

        contributions = pd.DataFrame({
            'feature': processed.columns,
            'value': processed.values[0],
            'shap_value': shap_vals[0]
        }).sort_values(
            'shap_value', key=abs, ascending=False
        )

        fraud_signals = contributions[
            contributions['shap_value'] > 0
        ].head(3)

        legit_signals = contributions[
            contributions['shap_value'] < 0
        ].head(3)

        result['explanation'] = {
            'base_rate': round(
                float(
                    components['explainer']\
                        .expected_value
                ), 4
            ),
            'top_fraud_signals': [
                {
                    'feature': row['feature'],
                    'value': round(
                        float(row['value']), 4
                    ),
                    'impact': round(
                        float(row['shap_value']), 4
                    )
                }
                for _, row in fraud_signals.iterrows()
            ],
            'top_legit_signals': [
                {
                    'feature': row['feature'],
                    'value': round(
                        float(row['value']), 4
                    ),
                    'impact': round(
                        float(row['shap_value']), 4
                    )
                }
                for _, row in legit_signals.iterrows()
            ]
        }

    return result