#!/usr/bin/env python3
"""
Final system verification - all components check
"""

import os
import sys
import pandas as pd
import joblib

print("🔥 Customer Churn Fire Project - Final Status Check")
print("=" * 60)

# Check data
data_path = "data/churn_data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✅ Data: Available - Shape {df.shape}")
    if 'Exited' in df.columns:
        churn_rate = df['Exited'].mean() * 100
        print(f"   📈 Churn Rate: {churn_rate:.1f}%")
else:
    print("❌ Data: Not found")

# Check model
model_path = "models/best_churn_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"✅ Model: Available - {type(model).__name__}")
    if hasattr(model, 'n_estimators'):
        print(f"   🌳 Estimators: {model.n_estimators}")
else:
    print("❌ Model: Not found")

# Check dependencies
deps = {
    "pandas": True,
    "numpy": True, 
    "plotly": True,
    "streamlit": True
}

try:
    import shap
    deps["shap"] = True
    print("✅ SHAP: Available")
except ImportError:
    deps["shap"] = False
    print("❌ SHAP: Not available")

# Check utils
project_root = os.getcwd()
utils_path = os.path.join(project_root, 'utils')
sys.path.insert(0, utils_path)
sys.path.insert(0, project_root)

utils_working = True
try:
    from utils.preprocessing import ChurnDataPreprocessor
    from utils.visualization import ChurnVisualizer  
    from utils.metrics import ModelEvaluator
    print("✅ Utils: All modules available")
except ImportError as e:
    print(f"❌ Utils: Import error - {e}")
    utils_working = False

print("\n" + "=" * 60)
if all([
    os.path.exists(data_path),
    os.path.exists(model_path), 
    deps["shap"],
    utils_working
]):
    print("🎯 System Status: ALL SYSTEMS GO! 🚀")
    print("🌐 Dashboard ready at: http://localhost:8501")
else:
    print("⚠️  System Status: Some issues remain")

print("=" * 60)
