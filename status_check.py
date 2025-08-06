"""
System status check for Customer Churn Analysis
"""
import os
import joblib
import pandas as pd

def check_system_status():
    """Check all system components"""
    status = {
        "data_available": False,
        "model_available": False,
        "data_info": {},
        "model_info": {}
    }
    
    # Check data
    data_path = "data/churn_data.csv"
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            status["data_available"] = True
            status["data_info"] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "churn_rate": df['Exited'].mean() * 100 if 'Exited' in df.columns else None
            }
        except Exception as e:
            status["data_info"]["error"] = str(e)
    
    # Check model
    model_path = "models/best_churn_model.joblib"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            status["model_available"] = True
            status["model_info"] = {
                "type": type(model).__name__,
                "features": getattr(model, 'n_features_in_', 'unknown')
            }
        except Exception as e:
            status["model_info"]["error"] = str(e)
    
    return status

if __name__ == "__main__":
    status = check_system_status()
    
    print("� Customer Churn Analysis - System Status")
    print("=" * 50)
    
    # Data status
    if status["data_available"]:
        print("✅ Data: Available")
        print(f"   📊 Shape: {status['data_info']['shape']}")
        if status['data_info']['churn_rate']:
            print(f"   📈 Churn Rate: {status['data_info']['churn_rate']:.1f}%")
    else:
        print("❌ Data: Not Available")
    
    # Model status  
    if status["model_available"]:
        print("✅ Model: Available")
        print(f"   🤖 Type: {status['model_info']['type']}")
        print(f"   📊 Features: {status['model_info']['features']}")
    else:
        print("❌ Model: Not Available")
    
    print("\n🎯 Overall Status:", "Ready!" if status["data_available"] and status["model_available"] else "Issues Found")
