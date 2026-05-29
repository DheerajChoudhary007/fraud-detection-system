# Create file: tests/test_performance.py

import requests
import time
import statistics

BASE_URL = "http://localhost:8000"

TRANSACTION = {
    "Time": 50000, "Amount": 150.0,
    "V1": -1.3598, "V2": -0.0728,
    "V3": 2.5364,  "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624,
    "V7": 0.2396,  "V8": 0.0987,
    "V9": 0.3638,  "V10": 0.0908,
    "V11": -0.5516,"V12": -0.6178,
    "V13": -0.9914,"V14": -0.3112,
    "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2076, "V18": 0.0258,
    "V19": 0.4039, "V20": 0.2514,
    "V21": -0.0183,"V22": 0.2778,
    "V23": -0.1105,"V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891,
    "V27": 0.1336, "V28": -0.0210
}

print("=== PERFORMANCE TEST ===\n")

# Test /predict (with SHAP)
print("Testing /predict (with SHAP)...")
times = []
for i in range(10):
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/predict",
        json=TRANSACTION
    )
    end = time.time()
    times.append((end - start) * 1000)

print(f"Average: {statistics.mean(times):.0f}ms")
print(f"Min:     {min(times):.0f}ms")
print(f"Max:     {max(times):.0f}ms")

# Test /predict/fast (without SHAP)
print("\nTesting /predict/fast (no SHAP)...")
times_fast = []
for i in range(10):
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/predict/fast",
        json=TRANSACTION
    )
    end = time.time()
    times_fast.append((end - start) * 1000)

print(f"Average: {statistics.mean(times_fast):.0f}ms")
print(f"Min:     {min(times_fast):.0f}ms")
print(f"Max:     {max(times_fast):.0f}ms")

print("\n=== PERFORMANCE SUMMARY ===")
print(f"With SHAP:    {statistics.mean(times):.0f}ms avg")
print(f"Without SHAP: {statistics.mean(times_fast):.0f}ms avg")
speedup = statistics.mean(times) / \
          statistics.mean(times_fast)
print(f"SHAP overhead: {speedup:.1f}x slower")

if statistics.mean(times) < 500:
    print("\n✅ API fast enough for production!")
else:
    print("\n⚠️ API slow — consider async SHAP")