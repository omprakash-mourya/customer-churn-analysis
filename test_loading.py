"""
Test script to verify model and data loading
"""
import os
import pandas as pd
import joblib

print("🔍 Testing model and data loading...")

# Test data loading
data_path = "data/churn_data.csv"
print(f"📁 Checking data file: {data_path}")
if os.path.exists(data_path):
    print("✅ Data file exists")
    df = pd.read_csv(data_path)
    print(f"📊 Data shape: {df.shape}")
    print(f"🎯 Columns: {list(df.columns)}")
    if 'Exited' in df.columns:
        churn_rate = df['Exited'].mean() * 100
        print(f"📈 Churn rate: {churn_rate:.1f}%")
else:
    print("❌ Data file not found")

# Test model loading
model_path = "models/best_churn_model.joblib"  
print(f"\n🤖 Checking model file: {model_path}")
if os.path.exists(model_path):
    print("✅ Model file exists")
    try:
        model = joblib.load(model_path)
        print(f"🎯 Model type: {type(model).__name__}")
        print(f"📊 Model features: {model.n_features_in_}")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
else:
    print("❌ Model file not found")

print("\n🎉 Test completed!")
