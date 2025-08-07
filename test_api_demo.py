"""
Demo script to test the FastAPI Customer Churn Prediction API
This shows you how to use all the available endpoints
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_model_info():
    """Test the model info endpoint"""
    print("📊 Testing Model Info...")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    print(f"Model Info: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_single_prediction():
    """Test single customer prediction"""
    print("🎯 Testing Single Prediction...")
    
    # Sample customer data
    customer_data = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 35,
        "Tenure": 5,
        "Balance": 75000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Churn Probability: {result['churn_probability']:.4f}")
        print(f"Confidence: {result['confidence']}")
        print(f"Risk Category: {result['risk_category']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_batch_prediction():
    """Test batch prediction with multiple customers"""
    print("📦 Testing Batch Prediction...")
    
    # Sample batch data
    batch_data = {
        "customers": [
            {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 5,
                "Balance": 75000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0
            },
            {
                "CreditScore": 800,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 45,
                "Tenure": 8,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results['predictions'])} customers")
        for i, pred in enumerate(results['predictions']):
            print(f"Customer {i+1}: {pred['prediction']} (Prob: {pred['churn_probability']:.4f}, {pred['confidence']})")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_drift_detection():
    """Test data drift detection"""
    print("🔄 Testing Data Drift Detection...")
    
    # Sample data for drift detection (using proper format)
    new_data = {
        "new_data": [
            {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 5,
                "Balance": 75000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0
            },
            {
                "CreditScore": 800,
                "Geography": "Germany",
                "Gender": "Male", 
                "Age": 45,
                "Tenure": 8,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/check-drift", json=new_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Drift Detected: {result['drift_detected']}")
        print(f"Drift Score (p-value): {result['drift_score']:.6f}")
        print(f"Statistical Test: {result['statistical_test']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def run_complete_demo():
    """Run complete API demonstration"""
    print("🚀 Customer Churn Prediction API Demo")
    print("=" * 50)
    
    try:
        # Test all endpoints
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_drift_detection()
        
        print("✅ All API tests completed successfully!")
        print(f"📋 API Documentation available at: {BASE_URL}/docs")
        print(f"📊 Interactive API docs at: {BASE_URL}/redoc")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("Make sure the FastAPI server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_complete_demo()
