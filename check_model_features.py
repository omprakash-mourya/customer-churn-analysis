#!/usr/bin/env python3
"""Check model features"""

import joblib
import os

try:
    model = joblib.load('./models/best_churn_model.joblib')
    print(f"Model type: {type(model)}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Features expected: {list(model.feature_names_in_)}")
    else:
        print("Model doesn't have feature_names_in_ attribute")
        
    # Also check if there are any feature files
    if os.path.exists('./models/'):
        files = os.listdir('./models/')
        print(f"Files in models directory: {files}")
        
except Exception as e:
    print(f"Error loading model: {e}")
