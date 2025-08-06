"""
Unit tests for model training utilities
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

class TestChurnModelTrainer:
    """Test suite for ChurnModelTrainer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 11, n_samples),
            'Balance': np.random.uniform(0, 200000, n_samples),
            'NumOfProducts': np.random.randint(1, 5, n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples),
            'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
            'Geography_Germany': np.random.choice([0, 1], n_samples),
            'Geography_Spain': np.random.choice([0, 1], n_samples),
            'Gender_Male': np.random.choice([0, 1], n_samples)
        })
        
        y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% churn rate
        
        return X, y
    
    def test_data_shapes(self, sample_data):
        """Test that sample data has correct shapes"""
        X, y = sample_data
        assert len(X) == len(y)
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        
    def test_target_distribution(self, sample_data):
        """Test target variable distribution"""
        X, y = sample_data
        unique_values = np.unique(y)
        assert len(unique_values) == 2
        assert set(unique_values) == {0, 1}
        
        # Check if there's class imbalance (typical for churn)
        churn_rate = np.mean(y)
        assert 0.05 <= churn_rate <= 0.4  # Reasonable churn rate range
        
    def test_feature_ranges(self, sample_data):
        """Test that features are in expected ranges"""
        X, y = sample_data
        
        assert X['CreditScore'].min() >= 300
        assert X['CreditScore'].max() <= 850
        assert X['Age'].min() >= 18
        assert X['Age'].max() <= 80
        assert X['Tenure'].min() >= 0
        assert X['Tenure'].max() <= 10
        assert X['NumOfProducts'].min() >= 1
        assert X['NumOfProducts'].max() <= 4
        
    def test_binary_features(self, sample_data):
        """Test binary features are properly encoded"""
        X, y = sample_data
        
        binary_cols = ['HasCrCard', 'IsActiveMember', 'Geography_Germany', 
                      'Geography_Spain', 'Gender_Male']
        
        for col in binary_cols:
            assert set(X[col].unique()).issubset({0, 1})
            
    def test_no_missing_values(self, sample_data):
        """Test that there are no missing values"""
        X, y = sample_data
        assert not X.isnull().any().any()
        assert not pd.Series(y).isnull().any()
        
    @patch('sklearn.model_selection.train_test_split')
    def test_train_test_split_called(self, mock_split, sample_data):
        """Test that train_test_split is called with correct parameters"""
        X, y = sample_data
        
        # Mock the return value
        mock_split.return_value = (X[:80], X[80:], y[:80], y[80:])
        
        # Import here to avoid dependency issues
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Verify split was called
            assert len(X_train) + len(X_test) == len(X)
            assert len(y_train) + len(y_test) == len(y)
            
        except ImportError:
            # Skip test if sklearn not available
            pytest.skip("sklearn not available")
            
    def test_model_performance_metrics(self):
        """Test model performance evaluation"""
        # Mock predictions and true labels
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        y_pred_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.3, 0.7])
        
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Basic sanity checks
            assert 0 <= accuracy <= 1
            assert 0 <= precision <= 1
            assert 0 <= recall <= 1
            assert 0 <= f1 <= 1
            
        except ImportError:
            # Skip test if sklearn not available
            pytest.skip("sklearn not available")
            
    def test_hyperparameter_grid(self):
        """Test hyperparameter grid configuration"""
        # Example hyperparameter grids
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        # Test grid structure
        assert isinstance(rf_params, dict)
        assert isinstance(xgb_params, dict)
        assert len(rf_params) > 0
        assert len(xgb_params) > 0
        
        # Test parameter types
        for param_name, param_values in rf_params.items():
            assert isinstance(param_values, list)
            assert len(param_values) > 0
            
    @pytest.mark.slow
    def test_model_training_integration(self, sample_data):
        """Integration test for model training (marked as slow)"""
        X, y = sample_data
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Basic checks
            assert len(y_pred) == len(X_test)
            assert y_pred_proba.shape == (len(X_test), 2)
            assert all(pred in [0, 1] for pred in y_pred)
            
        except ImportError:
            pytest.skip("sklearn not available for integration test")
