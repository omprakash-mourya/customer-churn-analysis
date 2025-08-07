"""
Test the complete integration between Streamlit and FastAPI
"""
import requests
import json

# Test data - 15 features as expected by the model
test_features = [
    650.0,    # CreditScore
    1.0,      # Geography_France  
    0.0,      # Geography_Germany
    0.0,      # Geography_Spain
    0.0,      # Gender_Male (Female)
    35.0,     # Age
    5.0,      # Tenure
    75000.0,  # Balance
    2.0,      # NumOfProducts
    1.0,      # HasCrCard
    1.0,      # IsActiveMember
    50000.0,  # EstimatedSalary
    115.38,   # CreditUtilization (Balance/CreditScore = 75000/650)
    4.0,      # InteractionScore (NumOfProducts + HasCrCard + IsActiveMember = 2+1+1)
    1.5       # BalanceToSalaryRatio (Balance/EstimatedSalary = 75000/50000)
]

print("Testing FastAPI prediction endpoint...")
print(f"Sending {len(test_features)} features: {test_features}")

try:
    payload = {"features": test_features}
    response = requests.post("http://127.0.0.1:8004/predict/simple", json=payload, timeout=10)
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            print("✅ SUCCESS: Prediction working!")
            print(f"   Prediction: {result.get('prediction')}")
            print(f"   Churn Probability: {result.get('churn_probability'):.2%}")
            print(f"   Confidence: {result.get('confidence'):.2f}")
        else:
            print(f"❌ ERROR: {result.get('error')}")
    else:
        print(f"❌ HTTP ERROR: {response.status_code}")
        
except Exception as e:
    print(f"❌ CONNECTION ERROR: {e}")

print("\nTesting health endpoint...")
try:
    response = requests.get("http://127.0.0.1:8004/health", timeout=5)
    print(f"Health check: {response.json()}")
except Exception as e:
    print(f"Health check failed: {e}")
