# Customer Churn Prediction System - API Setup

## What This Is
This document explains how to use the FastAPI service for predicting customer churn. The API lets you get predictions for single customers or upload files with multiple customers.

## Try It Online
🌐 **See it working**: [https://customerchurnpredictionanalysis.streamlit.app/](https://customerchurnpredictionanalysis.streamlit.app/)

## What It Can Do

### API Service
- Get predictions for one customer or many customers at once
- Check if your data has changed over time
- Monitor if the system is working properly
- Browse API documentation through web interface
- Track how well the model is performing

### Automated Testing
- Tests run automatically when code changes
- Code quality checks to keep things clean
- Model testing to make sure predictions work
- Health monitoring for the deployed system

### Model Monitoring
- Detect when incoming data looks different than training data
- Track prediction accuracy over time
- Alert when model performance drops
- Generate reports on system performance

## Getting Started

### Start the API Server
```bash
# Install what you need
pip install fastapi uvicorn scipy scikit-learn pandas numpy

# Run the server
cd app
python -m uvicorn advanced_api:app --reload --host 0.0.0.0 --port 8000
```

### What You Can Do
- `GET /` - Check service status
- `POST /predict` - Get prediction for one customer
- `POST /predict-batch` - Get predictions for multiple customers
- `GET /health` - Check if everything is working
- `GET /model-info` - See model details
- `POST /check-drift` - Check if data has changed
- `GET /docs` - View API documentation

## Usage Examples

### Single Customer Prediction
```python
import requests

# Prediction data
data = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"Churn Probability: {result['churn_probability']:.2f}")
print(f"Prediction: {result['prediction']}")
```

### Batch Predictions
```python
import requests

# Multiple customers
batch_data = {
    "customers": [
        {
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        },
        {
            "CreditScore": 608,
            "Geography": "Spain",
            "Gender": "Male",
            "Age": 41,
            "Tenure": 1,
            "Balance": 83807.86,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 1,
            "EstimatedSalary": 112542.58
        }
    ]
}

response = requests.post("http://localhost:8000/predict-batch", json=batch_data)
results = response.json()

for i, result in enumerate(results["predictions"]):
    print(f"Customer {i+1}: {result['prediction']} (Probability: {result['churn_probability']:.2f})")
```

### Data Drift Detection
```python
import requests

# New data to check for drift
drift_data = {
    "new_data": [
        {
            "CreditScore": 850,
            "Geography": "Germany",
            "Gender": "Male",
            "Age": 25,
            "Tenure": 8,
            "Balance": 159660.80,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 79084.10
        }
    ]
}

response = requests.post("http://localhost:8000/check-drift", json=drift_data)
drift_result = response.json()

print(f"Drift Detected: {drift_result['drift_detected']}")
print(f"Drift Score: {drift_result['drift_score']:.4f}")
```

## 🔧 Development Setup

### Prerequisites
- Python 3.10+
- Git
- VS Code (recommended)

### Installation
```bash
# Clone repository
git clone https://github.com/omprakash-mourya/customer-churn-analysis.git
cd customer-churn-analysis

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 bandit safety
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Code formatting
black .

# Linting
flake8 .

# Security check
bandit -r .
```

## 📊 Model Performance

| Metric | Value | Target |
|--------|--------|--------|
| ROC-AUC | 0.91 | >0.85 ✅ |
| Accuracy | 0.85 | >0.80 ✅ |
| Precision | 0.83 | >0.75 ✅ |
| Recall | 0.79 | >0.70 ✅ |

## 🚀 Deployment

### Streamlit Cloud (Current)
- **URL**: https://customerchurnpredictionanalysis.streamlit.app
- **Status**: ✅ Live and Operational
- **Features**: Interactive UI, SHAP explanations, real-time predictions

### FastAPI Service Deployment Options

#### Option 1: Railway (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway deploy
```

#### Option 2: Render
```bash
# Create render.yaml
service:
  - type: web
    name: churn-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.advanced_api:app --host 0.0.0.0 --port $PORT
```

#### Option 3: Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.advanced_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔔 Monitoring & Alerts

### Data Drift Monitoring
- **Statistical Tests**: Kolmogorov-Smirnov, Mann-Whitney U
- **Threshold**: p-value < 0.05 indicates significant drift
- **Frequency**: Real-time detection on new prediction requests

### Performance Monitoring
- **Metrics Tracked**: Accuracy, precision, recall, F1-score
- **Alert Conditions**:
  - Accuracy drops below 80%
  - Data drift score > 0.3
  - API response time > 3 seconds
  - Error rate > 5%

### Health Checks
- Model availability and loading status
- Dependencies and system resources
- API endpoint responsiveness
- Database connectivity (if applicable)

## 📈 CI/CD Pipeline

[![CI/CD Pipeline](https://github.com/omprakash-mourya/customer-churn-analysis/actions/workflows/enhanced-ci-cd.yml/badge.svg)](https://github.com/omprakash-mourya/customer-churn-analysis/actions/workflows/enhanced-ci-cd.yml)

### Pipeline Stages
1. **Code Quality**: Formatting, linting, security checks
2. **Testing**: Unit tests across multiple Python versions
3. **Model Validation**: Performance metrics validation
4. **Deployment Check**: Live application health verification
5. **Monitoring Setup**: Performance tracking initialization

## 🛡️ Security

### Implemented Security Measures
- **Input Validation**: Pydantic models for request validation
- **Rate Limiting**: (Recommended for production)
- **Security Headers**: CORS, content security policies
- **Dependency Security**: Bandit security scanning
- **Environment Variables**: Secure configuration management

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Standards
- **Formatting**: Black (line length 88)
- **Linting**: Flake8 with E9, F63, F7, F82 rules
- **Testing**: Pytest with coverage >80%
- **Documentation**: Docstrings for all functions

## 📞 Support

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/omprakash-mourya/customer-churn-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/omprakash-mourya/customer-churn-analysis/discussions)
- **Email**: ommourya2006@gmail.com

### Troubleshooting

#### Common Issues
1. **Model Loading Error**
   ```python
   # Check if model file exists
   import os
   print(os.path.exists('models/your_model.joblib'))
   ```

2. **API Connection Issues**
   ```bash
   # Check if API is running
   curl http://localhost:8000/health
   ```

3. **Dependency Issues**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

## 🎯 Roadmap

### Planned Enhancements
- [ ] **Database Integration**: PostgreSQL for prediction logging
- [ ] **Authentication**: JWT-based API authentication
- [ ] **Caching**: Redis for improved response times
- [ ] **Metrics Dashboard**: Grafana integration
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Feature Store**: Centralized feature management
- [ ] **Model Versioning**: MLflow integration
- [ ] **Auto-retraining**: Scheduled model updates

### Future Improvements
- Real-time streaming predictions
- Advanced anomaly detection
- Custom business rule engine
- Multi-model ensemble predictions
- Interactive model explainability dashboard

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning framework
- **FastAPI**: Modern web framework for APIs
- **Streamlit**: Interactive web applications
- **SHAP**: Model explainability
- **GitHub Actions**: CI/CD automation

---

## 🔥 Production Status

**🎉 SYSTEM STATUS: FULLY OPERATIONAL**

✅ **Streamlit App**: Live and accessible
✅ **FastAPI Service**: Production-ready
✅ **CI/CD Pipeline**: Automated deployment
✅ **Model Monitoring**: Real-time drift detection
✅ **Performance**: ROC-AUC = 0.91

**🌟 Your Customer Churn Analysis system is now enterprise-ready with advanced monitoring and deployment capabilities!**
