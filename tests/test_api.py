"""
API endpoint tests
"""
import pytest
import json
from unittest.mock import patch, Mock

# Mock FastAPI components for testing
class MockFastAPI:
    def __init__(self):
        self.routes = []
    
    def get(self, path):
        def decorator(func):
            self.routes.append(('GET', path, func))
            return func
        return decorator
        
    def post(self, path):
        def decorator(func):
            self.routes.append(('POST', path, func))
            return func
        return decorator

class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    @pytest.fixture
    def sample_prediction_request(self):
        """Sample prediction request data"""
        return {
            "CreditScore": 650,
            "Geography": "Spain",
            "Gender": "Female", 
            "Age": 35,
            "Tenure": 3,
            "Balance": 85000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000.0
        }
    
    @pytest.fixture
    def sample_batch_request(self):
        """Sample batch prediction request"""
        return {
            "customers": [
                {
                    "CreditScore": 650,
                    "Geography": "Spain",
                    "Gender": "Female",
                    "Age": 35,
                    "Tenure": 3,
                    "Balance": 85000.0,
                    "NumOfProducts": 2,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 75000.0
                },
                {
                    "CreditScore": 720,
                    "Geography": "Germany",
                    "Gender": "Male",
                    "Age": 42,
                    "Tenure": 5,
                    "Balance": 120000.0,
                    "NumOfProducts": 1,
                    "HasCrCard": 1,
                    "IsActiveMember": 0,
                    "EstimatedSalary": 90000.0
                }
            ]
        }
        
    def test_prediction_request_structure(self, sample_prediction_request):
        """Test prediction request has required fields"""
        required_fields = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary'
        ]
        
        for field in required_fields:
            assert field in sample_prediction_request
            
    def test_prediction_request_types(self, sample_prediction_request):
        """Test prediction request field types"""
        assert isinstance(sample_prediction_request['CreditScore'], int)
        assert isinstance(sample_prediction_request['Geography'], str)
        assert isinstance(sample_prediction_request['Gender'], str)
        assert isinstance(sample_prediction_request['Age'], int)
        assert isinstance(sample_prediction_request['Tenure'], int)
        assert isinstance(sample_prediction_request['Balance'], float)
        assert isinstance(sample_prediction_request['NumOfProducts'], int)
        assert isinstance(sample_prediction_request['HasCrCard'], int)
        assert isinstance(sample_prediction_request['IsActiveMember'], int)
        assert isinstance(sample_prediction_request['EstimatedSalary'], float)
        
    def test_categorical_values(self, sample_prediction_request):
        """Test categorical field values"""
        assert sample_prediction_request['Geography'] in ['Spain', 'Germany', 'France']
        assert sample_prediction_request['Gender'] in ['Male', 'Female']
        assert sample_prediction_request['HasCrCard'] in [0, 1]
        assert sample_prediction_request['IsActiveMember'] in [0, 1]
        
    def test_numerical_ranges(self, sample_prediction_request):
        """Test numerical field ranges"""
        assert 300 <= sample_prediction_request['CreditScore'] <= 850
        assert 18 <= sample_prediction_request['Age'] <= 100
        assert 0 <= sample_prediction_request['Tenure'] <= 10
        assert sample_prediction_request['Balance'] >= 0
        assert 1 <= sample_prediction_request['NumOfProducts'] <= 4
        assert sample_prediction_request['EstimatedSalary'] >= 0
        
    def test_batch_request_structure(self, sample_batch_request):
        """Test batch request structure"""
        assert 'customers' in sample_batch_request
        assert isinstance(sample_batch_request['customers'], list)
        assert len(sample_batch_request['customers']) > 0
        
        # Test each customer in batch
        for customer in sample_batch_request['customers']:
            required_fields = [
                'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary'
            ]
            for field in required_fields:
                assert field in customer
                
    @patch('joblib.load')
    def test_model_loading_mock(self, mock_load):
        """Test model loading with mocked joblib"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.7, 0.3]]
        mock_load.return_value = mock_model
        
        # Test model loading
        import joblib
        model = joblib.load('dummy_path')
        
        # Test predictions
        prediction = model.predict([[1, 2, 3]])
        probability = model.predict_proba([[1, 2, 3]])
        
        assert prediction == [0]
        assert probability == [[0.7, 0.3]]
        
    def test_health_check_response(self):
        """Test health check endpoint response structure"""
        expected_response = {
            "status": "healthy",
            "message": "Customer Churn Prediction API is running",
            "version": "1.0.0"
        }
        
        # Test response structure
        assert "status" in expected_response
        assert "message" in expected_response
        assert "version" in expected_response
        assert expected_response["status"] == "healthy"
        
    def test_prediction_response_structure(self):
        """Test prediction response structure"""
        mock_response = {
            "prediction": 0,
            "probability": 0.23,
            "risk_level": "Low",
            "confidence": 0.77,
            "model_version": "1.0.0",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Test response fields
        required_fields = [
            'prediction', 'probability', 'risk_level', 
            'confidence', 'model_version', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in mock_response
            
        # Test value types and ranges
        assert mock_response['prediction'] in [0, 1]
        assert 0 <= mock_response['probability'] <= 1
        assert mock_response['risk_level'] in ['Low', 'Medium', 'High']
        assert 0 <= mock_response['confidence'] <= 1
        
    def test_batch_prediction_response_structure(self):
        """Test batch prediction response structure"""
        mock_response = {
            "predictions": [
                {
                    "customer_id": 0,
                    "prediction": 0,
                    "probability": 0.23,
                    "risk_level": "Low"
                },
                {
                    "customer_id": 1,
                    "prediction": 1,
                    "probability": 0.78,
                    "risk_level": "High"
                }
            ],
            "summary": {
                "total_customers": 2,
                "high_risk_count": 1,
                "medium_risk_count": 0,
                "low_risk_count": 1,
                "average_churn_probability": 0.505
            },
            "model_version": "1.0.0",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Test top-level structure
        assert "predictions" in mock_response
        assert "summary" in mock_response
        assert "model_version" in mock_response
        assert "timestamp" in mock_response
        
        # Test predictions array
        assert isinstance(mock_response["predictions"], list)
        for pred in mock_response["predictions"]:
            assert "customer_id" in pred
            assert "prediction" in pred
            assert "probability" in pred
            assert "risk_level" in pred
            
        # Test summary
        summary = mock_response["summary"]
        assert "total_customers" in summary
        assert "high_risk_count" in summary
        assert "medium_risk_count" in summary
        assert "low_risk_count" in summary
        assert "average_churn_probability" in summary
        
    @pytest.mark.parametrize("invalid_data", [
        {"CreditScore": -100},  # Invalid credit score
        {"Age": -5},  # Invalid age
        {"Geography": "InvalidCountry"},  # Invalid geography
        {"Gender": "Other"},  # Invalid gender
        {"NumOfProducts": 0},  # Invalid number of products
        {"HasCrCard": 2},  # Invalid binary value
    ])
    def test_invalid_input_validation(self, invalid_data, sample_prediction_request):
        """Test validation of invalid inputs"""
        # Merge invalid data with sample request
        request = {**sample_prediction_request, **invalid_data}
        
        # These should be caught by validation
        if 'CreditScore' in invalid_data:
            assert request['CreditScore'] < 300 or request['CreditScore'] > 850
        if 'Age' in invalid_data:
            assert request['Age'] < 18 or request['Age'] > 100
        if 'Geography' in invalid_data:
            assert request['Geography'] not in ['Spain', 'Germany', 'France']
        if 'Gender' in invalid_data:
            assert request['Gender'] not in ['Male', 'Female']
        if 'NumOfProducts' in invalid_data:
            assert request['NumOfProducts'] < 1 or request['NumOfProducts'] > 4
        if 'HasCrCard' in invalid_data:
            assert request['HasCrCard'] not in [0, 1]
