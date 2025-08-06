#!/usr/bin/env python3
"""
🧪 Local API Testing Script
Tests the enhanced FastAPI service locally before deployment
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🎯 Testing Single Prediction...")
    
    # Test data
    test_customer = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_customer)
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction working!")
            print(f"   Customer Profile: Age={test_customer['Age']}, Geography={test_customer['Geography']}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Churn Probability: {result['churn_probability']:.3f}")
            print(f"   Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return False

def test_batch_predictions():
    """Test batch prediction endpoint"""
    print("\n📊 Testing Batch Predictions...")
    
    # Multiple test customers
    batch_data = {
        "customers": [
            {
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88
            },
            {
                "CreditScore": 608,
                "Geography": "Spain",
                "Gender": "Male",
                "Age": 41,
                "Tenure": 1,
                "Balance": 83807.86,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 1,
                "EstimatedSalary": 112542.58
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction working!")
            print(f"   Processed {len(result['predictions'])} customers")
            for i, pred in enumerate(result['predictions']):
                print(f"   Customer {i+1}: {pred['prediction']} (Prob: {pred['churn_probability']:.3f})")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_data_drift_detection():
    """Test data drift detection endpoint"""
    print("\n📈 Testing Data Drift Detection...")
    
    # Test data for drift detection
    drift_data = {
        "new_data": [
            {
                "CreditScore": 850,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 25,
                "Tenure": 8,
                "Balance": 159660.80,
                "NumOfProducts": 3,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 79084.10
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/check-drift", json=drift_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Drift detection working!")
            print(f"   Drift Detected: {result['drift_detected']}")
            print(f"   Drift Score: {result['drift_score']:.4f}")
            print(f"   Statistical Test: {result['statistical_test']}")
            return True
        else:
            print(f"❌ Drift detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Drift detection error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n📋 Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model info working!")
            print(f"   Model: {result['model_name']}")
            print(f"   Version: {result['version']}")
            print(f"   Features: {len(result['feature_names'])} features")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting Local API Tests...")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for API server to be ready...")
    time.sleep(2)
    
    tests = [
        test_health_endpoint,
        test_model_info,
        test_single_prediction,
        test_batch_predictions,
        test_data_drift_detection
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED! API is ready for deployment.")
        print("\n🌐 Your FastAPI service is available at:")
        print(f"   • API Docs: {BASE_URL}/docs")
        print(f"   • Health: {BASE_URL}/health")
        print(f"   • Predictions: {BASE_URL}/predict")
        return True
    else:
        print("\n❌ Some tests failed. Please check the API service.")
        return False

if __name__ == "__main__":
    main()
