"""
Working FastAPI for Customer Churn Prediction
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import os
import uvicorn

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Request model  
class PredictionRequest(BaseModel):
    features: List[float]

# Load model once at startup
MODEL = None
model_path = r"C:\Users\ommou\OneDrive\Desktop\Custommer_churn_analysis\CustomerChurnFireProject\models\best_churn_model.joblib"

try:
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        print(f"SUCCESS: Model loaded from {model_path}")
    else:
        print(f"ERROR: Model file not found at {model_path}")
except Exception as e:
    print(f"ERROR loading model: {e}")

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "model_loaded": MODEL is not None,
        "version": "2.0"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_status": "loaded" if MODEL is not None else "not loaded"
    }

@app.post("/predict/simple")
def predict_simple(request: PredictionRequest):
    """
    Simple prediction with list of preprocessed features.
    Expects 15 features in this order:
    0: CreditScore, 1: Geography_France, 2: Geography_Germany, 3: Geography_Spain, 
    4: Gender_Male, 5: Age, 6: Tenure, 7: Balance, 8: NumOfProducts, 
    9: HasCrCard, 10: IsActiveMember, 11: EstimatedSalary, 12: CreditUtilization,
    13: InteractionScore, 14: BalanceToSalaryRatio
    """
    features = request.features
    
    if MODEL is None:
        return {"error": "Model not loaded", "model_path": model_path}
    
    try:
        # Check number of features - the model expects 15 features
        expected_features = 15
        if len(features) != expected_features:
            return {"error": f"Expected {expected_features} features, got {len(features)}. Model was trained with feature engineering."}
        
        # Make prediction
        features_array = np.array([features])
        prediction = MODEL.predict(features_array)[0]
        probability = MODEL.predict_proba(features_array)[0]
        
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability[1]),
            "confidence": float(max(probability)),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
