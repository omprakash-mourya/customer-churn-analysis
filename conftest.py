# Customer Churn Fire Project - Testing Configuration
import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        'CreditScore': 650,
        'Geography': 'Spain',
        'Gender': 'Female',
        'Age': 35,
        'Tenure': 3,
        'Balance': 85000.0,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 75000.0
    }

@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing"""
    import pandas as pd
    
    return pd.DataFrame([
        {
            'CreditScore': 650,
            'Geography': 'Spain', 
            'Gender': 'Female',
            'Age': 35,
            'Tenure': 3,
            'Balance': 85000.0,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 75000.0
        },
        {
            'CreditScore': 720,
            'Geography': 'Germany',
            'Gender': 'Male', 
            'Age': 42,
            'Tenure': 5,
            'Balance': 120000.0,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 90000.0
        }
    ])

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir

# Test configuration
def pytest_configure(config):
    """Pytest configuration"""
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
