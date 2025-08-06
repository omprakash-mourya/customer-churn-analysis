"""
Simple mock model creator for demonstration purposes
"""
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Add current directory to path
sys.path.append('.')

def create_mock_model():
    """Create a simple mock model with sample data"""
    print("🤖 Creating mock churn prediction model...")
    
    # Generate simple mock data
    np.random.seed(42)
    n_samples = 1000
    
    # Mock features
    X = np.random.rand(n_samples, 8)  # 8 features
    
    # Mock target (churn) - biased based on features
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.normal(0, 0.2, n_samples)) > 0.5
    y = y.astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train simple Random Forest
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_churn_model.joblib')
    print(f"✅ Model saved to models/best_churn_model.joblib")
    
    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"📊 Model accuracy: {accuracy:.3f}")
    
    return model

if __name__ == "__main__":
    create_mock_model()
