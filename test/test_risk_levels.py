# Create file: tests/test_risk_levels.py
import requests

BASE_URL = "http://localhost:8000"

print("=== RISK LEVEL TESTS ===\n")

# Test different transactions
test_cases = [
    ("Very obvious fraud", {
        "Time": 10000, "Amount": 1.0,
        "V1": -3.0435, "V2": -3.1572,
        "V3": 1.0887,  "V4": 2.2886,
        "V5": 1.3598,  "V6": -1.0197,
        "V7": -0.7238, "V8": 0.1654,
        "V9": -0.4516, "V10": -2.5579,
        "V11": 1.6420, "V12": -2.5921,
        "V13": 0.5202, "V14": -3.4742,
        "V15": 0.9434, "V16": -2.2246,
        "V17": -3.5495,"V18": -0.7593,
        "V19": 0.4126, "V20": 0.0610,
        "V21": 0.6603, "V22": 0.5543,
        "V23": -0.2060,"V24": -0.2516,
        "V25": -0.5329,"V26": 0.0324,
        "V27": 0.2432, "V28": 0.0977
    }),
    ("Normal legit", {
        "Time": 50000, "Amount": 50.0,
        "V1": 1.1918,  "V2": 0.2661,
        "V3": 0.1665,  "V4": 0.4481,
        "V5": 0.0600,  "V6": -0.0824,
        "V7": -0.0788, "V8": 0.0851,
        "V9": -0.2552, "V10": -0.1669,
        "V11": 1.6127, "V12": 1.0650,
        "V13": 0.4890, "V14": -0.1437,
        "V15": 0.6355, "V16": 0.4639,
        "V17": -0.1140,"V18": -0.1837,
        "V19": -0.1459,"V20": -0.0691,
        "V21": -0.2257,"V22": -0.6386,
        "V23": -0.0801,"V24": 0.4173,
        "V25": 0.4126, "V26": 0.3943,
        "V27": 0.1008, "V28": 0.0579
    })
]

for name, transaction in test_cases:
    response = requests.post(
        f"{BASE_URL}/predict",
        json=transaction
    )
    data = response.json()

    print(f"Transaction: {name}")
    print(f"  Probability: {data['fraud_probability']}")
    print(f"  Is Fraud:    {data['is_fraud']}")
    print(f"  Risk Level:  {data['risk_level']}")
    print(f"  Top Signal:  "
          f"{data['explanation']['top_fraud_signals'][0]['feature']}")
    print()