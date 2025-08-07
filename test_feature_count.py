"""
Test the exact number of features expected by the trained model
"""
import joblib
import numpy as np

# Load the model
model = joblib.load("models/best_churn_model.joblib")

# Test with different feature counts
for n_features in [8, 10, 12, 15, 19, 20]:
    try:
        test_features = np.random.random((1, n_features))
        prediction = model.predict(test_features)
        print(f"✅ {n_features} features: SUCCESS")
        print(f"   Model expects exactly {n_features} features")
        break
    except Exception as e:
        print(f"❌ {n_features} features: {e}")

# Also check the model's feature count if available
try:
    if hasattr(model, 'n_features_in_'):
        print(f"\nModel's n_features_in_: {model.n_features_in_}")
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances length: {len(model.feature_importances_)}")
except Exception as e:
    print(f"Error checking model attributes: {e}")
