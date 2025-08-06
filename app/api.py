"""
FastAPI application for Customer Churn Prediction.

This API provides real-time churn prediction endpoints with comprehensive
model explainability using SHAP values.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os
import sys
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.preprocessing import ChurnDataPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time customer churn prediction with explainability",
    version="1.0.0"
)

# Global variables
MODEL = None
PREPROCESSOR = None
SHAP_EXPLAINER = None
FEATURE_NAMES = None

class CustomerFeatures(BaseModel):
    """Customer features for prediction."""
    CreditScore: float = Field(..., ge=300, le=850, description="Customer credit score")
    Geography: str = Field(..., description="Customer geography (France/Spain/Germany)")
    Gender: str = Field(..., description="Customer gender (Male/Female)")
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Tenure: int = Field(..., ge=0, le=20, description="Years with bank")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0/1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0/1)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    customers: List[CustomerFeatures]

class PredictionResponse(BaseModel):
    """Prediction response."""
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: int
    risk_level: str
    shap_values: Optional[Dict] = None
    feature_contributions: Optional[Dict] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    summary: Dict

@app.on_event("startup")
async def load_model():
    """Load the trained model and preprocessor on startup."""
    global MODEL, PREPROCESSOR, SHAP_EXPLAINER, FEATURE_NAMES
    
    try:
        # Load the best model
        model_path = "models/best_churn_model.joblib"
        if not os.path.exists(model_path):
            # Fallback to any available model
            model_files = [f for f in os.listdir("models/") if f.endswith('.joblib')]
            if model_files:
                model_path = os.path.join("models", model_files[0])
                print(f"⚠️ Best model not found, using: {model_path}")
            else:
                print("❌ No trained models found. Please run training first.")
                return
        
        MODEL = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        
        # Initialize preprocessor
        PREPROCESSOR = ChurnDataPreprocessor()
        
        # Load sample data to get feature names and initialize SHAP
        sample_data_path = "data/churn_data.csv"
        if os.path.exists(sample_data_path):
            # Load and preprocess sample data for SHAP background
            df_sample = pd.read_csv(sample_data_path).sample(100, random_state=42)
            
            # Clean and encode sample data
            df_sample = PREPROCESSOR.clean_data(df_sample)
            df_sample = PREPROCESSOR.feature_engineering(df_sample)
            df_sample = PREPROCESSOR.encode_categorical(df_sample, 'Exited')
            
            X_sample, _ = PREPROCESSOR.prepare_features_target(df_sample, 'Exited')
            FEATURE_NAMES = X_sample.columns.tolist()
            
            # Initialize SHAP explainer
            try:
                SHAP_EXPLAINER = shap.TreeExplainer(MODEL)
                print("✅ SHAP explainer initialized")
            except Exception as e:
                print(f"⚠️ SHAP explainer initialization failed: {e}")
                SHAP_EXPLAINER = None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        MODEL = None

def preprocess_customer_data(customer_data: CustomerFeatures) -> pd.DataFrame:
    """Preprocess customer data for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([customer_data.dict()])
    
    # Apply feature engineering
    df = PREPROCESSOR.feature_engineering(df)
    
    # Encode categorical variables
    df = PREPROCESSOR.encode_categorical(df, target_col=None)
    
    # Ensure all required features are present
    if FEATURE_NAMES:
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[FEATURE_NAMES]
    
    return df

def get_risk_level(probability: float) -> str:
    """Get risk level based on churn probability."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

def get_shap_explanation(customer_df: pd.DataFrame, customer_id: str = None) -> Dict:
    """Get SHAP explanation for prediction."""
    if SHAP_EXPLAINER is None:
        return None
    
    try:
        shap_values = SHAP_EXPLAINER.shap_values(customer_df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
        
        # Create feature contribution dictionary
        feature_contributions = {}
        for i, feature in enumerate(customer_df.columns):
            contribution = float(shap_values[0][i])
            feature_contributions[feature] = {
                'value': float(customer_df.iloc[0, i]),
                'shap_value': contribution,
                'contribution': 'positive' if contribution > 0 else 'negative'
            }
        
        # Sort by absolute SHAP value
        sorted_contributions = dict(
            sorted(feature_contributions.items(), 
                   key=lambda x: abs(x[1]['shap_value']), 
                   reverse=True)
        )
        
        return sorted_contributions
        
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "model_loaded": MODEL is not None,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_status": "loaded" if MODEL is not None else "not loaded",
        "shap_status": "available" if SHAP_EXPLAINER is not None else "not available"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures, explain: bool = True):
    """Predict churn probability for a single customer."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess customer data
        customer_df = preprocess_customer_data(customer)
        
        # Make prediction
        churn_probability = float(MODEL.predict_proba(customer_df)[0][1])
        churn_prediction = int(MODEL.predict(customer_df)[0])
        risk_level = get_risk_level(churn_probability)
        
        # Get SHAP explanation if requested
        shap_explanation = None
        if explain and SHAP_EXPLAINER is not None:
            shap_explanation = get_shap_explanation(customer_df)
        
        return PredictionResponse(
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            feature_contributions=shap_explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest, explain: bool = False):
    """Predict churn probability for multiple customers."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        high_risk_count = 0
        
        for i, customer in enumerate(request.customers):
            # Preprocess customer data
            customer_df = preprocess_customer_data(customer)
            
            # Make prediction
            churn_probability = float(MODEL.predict_proba(customer_df)[0][1])
            churn_prediction = int(MODEL.predict(customer_df)[0])
            risk_level = get_risk_level(churn_probability)
            
            if risk_level == "HIGH":
                high_risk_count += 1
            
            # Get SHAP explanation if requested (only for small batches)
            shap_explanation = None
            if explain and len(request.customers) <= 10 and SHAP_EXPLAINER is not None:
                shap_explanation = get_shap_explanation(customer_df, f"customer_{i}")
            
            predictions.append(PredictionResponse(
                customer_id=f"customer_{i}",
                churn_probability=churn_probability,
                churn_prediction=churn_prediction,
                risk_level=risk_level,
                feature_contributions=shap_explanation
            ))
        
        # Calculate summary statistics
        probabilities = [pred.churn_probability for pred in predictions]
        summary = {
            "total_customers": len(request.customers),
            "high_risk_customers": high_risk_count,
            "average_churn_probability": np.mean(probabilities),
            "max_churn_probability": np.max(probabilities),
            "min_churn_probability": np.min(probabilities)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...), explain: bool = False):
    """Predict churn from uploaded CSV file."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate required columns
        required_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                           'EstimatedSalary']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Process each row
        predictions = []
        for index, row in df.iterrows():
            customer_data = CustomerFeatures(**row[required_columns].to_dict())
            customer_df = preprocess_customer_data(customer_data)
            
            # Make prediction
            churn_probability = float(MODEL.predict_proba(customer_df)[0][1])
            churn_prediction = int(MODEL.predict(customer_df)[0])
            risk_level = get_risk_level(churn_probability)
            
            predictions.append({
                "customer_id": f"row_{index}",
                "churn_probability": churn_probability,
                "churn_prediction": churn_prediction,
                "risk_level": risk_level
            })
        
        # Calculate summary
        probabilities = [pred["churn_probability"] for pred in predictions]
        high_risk_count = sum(1 for pred in predictions if pred["risk_level"] == "HIGH")
        
        summary = {
            "total_customers": len(predictions),
            "high_risk_customers": high_risk_count,
            "average_churn_probability": np.mean(probabilities),
            "max_churn_probability": np.max(probabilities),
            "min_churn_probability": np.min(probabilities)
        }
        
        return {
            "predictions": predictions,
            "summary": summary,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = {
        "model_type": str(type(MODEL).__name__),
        "features_count": len(FEATURE_NAMES) if FEATURE_NAMES else None,
        "feature_names": FEATURE_NAMES,
        "shap_available": SHAP_EXPLAINER is not None
    }
    
    # Add model-specific information
    if hasattr(MODEL, 'feature_importances_'):
        importances = MODEL.feature_importances_.tolist() if FEATURE_NAMES else None
        if importances and FEATURE_NAMES:
            model_info["feature_importances"] = dict(zip(FEATURE_NAMES, importances))
    
    return model_info

@app.get("/features/importance")
async def get_feature_importance():
    """Get feature importance from the model."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not hasattr(MODEL, 'feature_importances_'):
        raise HTTPException(
            status_code=400, 
            detail="Feature importance not available for this model type"
        )
    
    if not FEATURE_NAMES:
        raise HTTPException(status_code=503, detail="Feature names not available")
    
    importances = MODEL.feature_importances_
    feature_importance = [
        {"feature": name, "importance": float(importance)}
        for name, importance in zip(FEATURE_NAMES, importances)
    ]
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return {"feature_importance": feature_importance}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
