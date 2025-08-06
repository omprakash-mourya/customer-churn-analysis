import os
import sys
import pandas as pd
import joblib

print("Customer Churn Analysis - System Status Check")
print("=" * 50)

# Check data
data_path = "data/churn_data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Data: Available - Shape {df.shape}")
    if 'Exited' in df.columns:
        churn_rate = df['Exited'].mean() * 100
        print(f"   Churn Rate: {churn_rate:.1f}%")
else:
    print("Data: Not found")

# Check model
model_path = "models/best_churn_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model: Available - {type(model).__name__}")
else:
    print("Model: Not found")

# Check SHAP
try:
    import shap
    print("SHAP: Available")
    shap_ok = True
except ImportError:
    print("SHAP: Not available")
    shap_ok = False

# Check utils
project_root = os.getcwd()
utils_path = os.path.join(project_root, 'utils')
sys.path.insert(0, utils_path)
sys.path.insert(0, project_root)

try:
    from utils.preprocessing import ChurnDataPreprocessor
    from utils.visualization import ChurnVisualizer  
    from utils.metrics import ModelEvaluator
    print("Utils: All modules available")
    utils_ok = True
except ImportError as e:
    print(f"Utils: Import error - {e}")
    utils_ok = False

print("\n" + "=" * 50)
if all([
    os.path.exists(data_path),
    os.path.exists(model_path), 
    shap_ok,
    utils_ok
]):
    print("System Status: ALL SYSTEMS GO!")
    print("Dashboard ready at: http://localhost:8501")
else:
    print("System Status: Some issues remain")
print("=" * 50)
