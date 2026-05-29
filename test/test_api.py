# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)

# ─────────────────────────────────────
# SAMPLE TRANSACTIONS FOR TESTING
# ─────────────────────────────────────

# Normal legit transaction
LEGIT_TRANSACTION = {
    "Time": 50000,
    "Amount": 50.0,
    "V1": 1.1918, "V2": 0.2661,
    "V3": 0.1665, "V4": 0.4481,
    "V5": 0.0600, "V6": -0.0824,
    "V7": -0.0788, "V8": 0.0851,
    "V9": -0.2552, "V10": -0.1669,
    "V11": 1.6127, "V12": 1.0650,
    "V13": 0.4890, "V14": -0.1437,
    "V15": 0.6355, "V16": 0.4639,
    "V17": -0.1140, "V18": -0.1837,
    "V19": -0.1459, "V20": -0.0691,
    "V21": -0.2257, "V22": -0.6386,
    "V23": -0.0801, "V24": 0.4173,
    "V25": 0.4126, "V26": 0.3943,
    "V27": 0.1008, "V28": 0.0579
}

# Suspicious fraud transaction
FRAUD_TRANSACTION = {
    "Time": 10000,
    "Amount": 1.0,
    "V1": -3.0435, "V2": -3.1572,
    "V3": 1.0887, "V4": 2.2886,
    "V5": 1.3598, "V6": -1.0197,
    "V7": -0.7238, "V8": 0.1654,
    "V9": -0.4516, "V10": -2.5579,
    "V11": 1.6420, "V12": -2.5921,
    "V13": 0.5202, "V14": -3.4742,
    "V15": 0.9434, "V16": -2.2246,
    "V17": -3.5495, "V18": -0.7593,
    "V19": 0.4126, "V20": 0.0610,
    "V21": 0.6603, "V22": 0.5543,
    "V23": -0.2060, "V24": -0.2516,
    "V25": -0.5329, "V26": 0.0324,
    "V27": 0.2432, "V28": 0.0977
}

# Missing fields transaction
INCOMPLETE_TRANSACTION = {
    "Time": 50000,
    "Amount": 100.0,
    "V1": -1.35
    # Missing V2 to V28
}

# Wrong data types
WRONG_TYPES_TRANSACTION = {
    "Time": "not_a_number",
    "Amount": 100.0,
    "V1": -1.35,
    "V2": -0.07,
    "V3": 2.53, "V4": 1.37,
    "V5": -0.33, "V6": 0.46,
    "V7": 0.23, "V8": 0.09,
    "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31,
    "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02,
    "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27,
    "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18,
    "V27": 0.13, "V28": -0.02
}

# Negative amount
NEGATIVE_AMOUNT = {**LEGIT_TRANSACTION,
                   "Amount": -100.0}

# Zero amount
ZERO_AMOUNT = {**LEGIT_TRANSACTION,
               "Amount": 0.0}

# Very large amount
LARGE_AMOUNT = {**LEGIT_TRANSACTION,
                "Amount": 999999.0}


# ─────────────────────────────────────
# GENERAL ENDPOINT TESTS
# ─────────────────────────────────────

class TestGeneral:

    def test_root(self):
        """Test welcome endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        print("✅ Root endpoint works!")

    def test_health(self):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] == True
        assert data["threshold"] == 0.5
        assert "auc_roc" in data
        print(f"✅ Health check: {data['status']}")

    def test_model_info(self):
        """Test model information"""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "XGBoost"
        assert data["auc_roc"] > 0.95
        assert data["threshold"] == 0.5
        assert data["fraud_recall"] > 0.80
        print(f"✅ Model info: AUC={data['auc_roc']}")


# ─────────────────────────────────────
# PREDICTION ENDPOINT TESTS
# ─────────────────────────────────────

class TestPrediction:

    def test_predict_legit(self):
        """Test prediction on legit transaction"""
        response = client.post(
            "/predict",
            json=LEGIT_TRANSACTION
        )
        assert response.status_code == 200
        data = response.json()

        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "threshold_used" in data
        assert "explanation" in data
        assert data["threshold_used"] == 0.5
        assert 0 <= data["fraud_probability"] <= 1
        assert data["risk_level"] in [
            "LOW", "MEDIUM", "HIGH"
        ]
        print(f"✅ Legit transaction: "
              f"prob={data['fraud_probability']}, "
              f"fraud={data['is_fraud']}")

    def test_predict_fraud(self):
        """Test prediction on fraud transaction"""
        response = client.post(
            "/predict",
            json=FRAUD_TRANSACTION
        )
        assert response.status_code == 200
        data = response.json()
        assert data["fraud_probability"] > 0.5
        print(f"✅ Fraud transaction: "
              f"prob={data['fraud_probability']}, "
              f"fraud={data['is_fraud']}")

    def test_predict_explanation(self):
        """Test SHAP explanation is returned"""
        response = client.post(
            "/predict",
            json=FRAUD_TRANSACTION
        )
        data = response.json()
        explanation = data["explanation"]

        assert "base_rate" in explanation
        assert "top_fraud_signals" in explanation
        assert "top_legit_signals" in explanation
        assert len(
            explanation["top_fraud_signals"]
        ) > 0

        signal = explanation["top_fraud_signals"][0]
        assert "feature" in signal
        assert "value" in signal
        assert "impact" in signal
        print(f"✅ SHAP explanation: "
              f"top feature = "
              f"{signal['feature']}, "
              f"impact = {signal['impact']}")

    def test_predict_threshold(self):
        """Test threshold is always 0.5"""
        response = client.post(
            "/predict",
            json=LEGIT_TRANSACTION
        )
        data = response.json()
        assert data["threshold_used"] == 0.5
        print("✅ Threshold correctly set to 0.5!")

    def test_predict_fast(self):
        """Test fast prediction (no SHAP)"""
        response = client.post(
            "/predict/fast",
            json=LEGIT_TRANSACTION
        )
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        print(f"✅ Fast predict works: "
              f"prob={data['fraud_probability']}")

    def test_predict_batch(self):
        """Test batch prediction"""
        batch = [LEGIT_TRANSACTION,
                 FRAUD_TRANSACTION,
                 LEGIT_TRANSACTION]
        response = client.post(
            "/predict/batch",
            json=batch
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert "fraud_count" in data
        assert len(data["results"]) == 3
        print(f"✅ Batch predict: "
              f"{data['total']} transactions, "
              f"{data['fraud_count']} fraud")


# ─────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────

class TestEdgeCases:

    def test_missing_fields(self):
        """Test missing required fields"""
        response = client.post(
            "/predict",
            json=INCOMPLETE_TRANSACTION
        )
        # Should return 422 validation error
        assert response.status_code == 422
        print("✅ Missing fields handled: 422 error")

    def test_wrong_data_types(self):
        """Test wrong data types"""
        response = client.post(
            "/predict",
            json=WRONG_TYPES_TRANSACTION
        )
        assert response.status_code == 422
        print("✅ Wrong types handled: 422 error")

    def test_negative_amount(self):
        """Test negative amount rejected"""
        response = client.post(
            "/predict",
            json=NEGATIVE_AMOUNT
        )
        # Should be rejected (amount >= 0)
        assert response.status_code == 422
        print("✅ Negative amount rejected!")

    def test_zero_amount(self):
        """Test zero amount accepted"""
        response = client.post(
            "/predict",
            json=ZERO_AMOUNT
        )
        assert response.status_code == 200
        print("✅ Zero amount accepted!")

    def test_large_amount(self):
        """Test very large amount"""
        response = client.post(
            "/predict",
            json=LARGE_AMOUNT
        )
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["fraud_probability"] <= 1
        print(f"✅ Large amount handled: "
              f"prob={data['fraud_probability']}")

    def test_batch_limit(self):
        """Test batch limit of 100"""
        # Create 101 transactions
        big_batch = [LEGIT_TRANSACTION] * 101
        response = client.post(
            "/predict/batch",
            json=big_batch
        )
        assert response.status_code == 400
        print("✅ Batch limit enforced: 400 error")

    def test_empty_batch(self):
        """Test empty batch"""
        response = client.post(
            "/predict/batch",
            json=[]
        )
        # Empty batch should still work
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        print("✅ Empty batch handled!")