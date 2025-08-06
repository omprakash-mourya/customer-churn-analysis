"""
Simple FastAPI test without complex dependencies
"""
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Customer Churn Prediction API - Simple", version="1.0.0")

# Load model once at startup
MODEL = None
if os.path.exists("models/best_churn_model.joblib"):
    try:
        MODEL = joblib.load("models/best_churn_model.joblib")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "model_loaded": MODEL is not None
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_status": "loaded" if MODEL is not None else "not loaded"
    }

@app.post("/predict/simple")
def predict_simple(features: list):
    """Simple prediction with list of features"""
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    try:
        # Ensure we have the right number of features
        if len(features) != 8:
            return {"error": f"Expected 8 features, got {len(features)}"}
        
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
