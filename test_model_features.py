import joblib
import numpy as np

# Load the model
model = joblib.load('models/best_churn_model.joblib')
print('Model type:', type(model))

# Test with 8 features
test_features_8 = np.array([[650, 1, 0, 35, 5, 75000, 2, 1]])
print('8 features test shape:', test_features_8.shape)
try:
    pred = model.predict(test_features_8)
    print('8 features prediction works:', pred)
except Exception as e:
    print('8 features error:', e)

# Test with 10 features  
test_features_10 = np.array([[650, 1, 0, 35, 5, 75000, 2, 1, 1, 50000]])
print('10 features test shape:', test_features_10.shape)
try:
    pred = model.predict(test_features_10)
    print('10 features prediction works:', pred)
except Exception as e:
    print('10 features error:', e)

# Test with 11 features (all including geography/gender encoded)
test_features_11 = np.array([[650, 0, 1, 0, 35, 5, 75000, 2, 1, 1, 50000]])
print('11 features test shape:', test_features_11.shape)
try:
    pred = model.predict(test_features_11)
    print('11 features prediction works:', pred)
except Exception as e:
    print('11 features error:', e)
