# API Deployment Guide

## Overview
This guide covers the deployment process for the Customer Churn Analysis API service with monitoring capabilities.

## Live Demo
🌐 **Access the application**: [https://customerchurnpredictionanalysis.streamlit.app/](https://customerchurnpredictionanalysis.streamlit.app/)

## Local Development

### Setup Instructions
```bash
# Navigate to project directory
cd c:\Users\ommou\OneDrive\Desktop\Custommer_churn_analysis\CustomerChurnFireProject

# Install additional dependencies for API
pip install fastapi uvicorn scipy python-multipart

# Start the FastAPI server
cd app
python -m uvicorn advanced_api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test API Endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test API documentation
# Open in browser: http://localhost:8000/docs
```

### 3. Verify CI/CD Pipeline
```bash
# Check GitHub Actions workflow
# Navigate to: https://github.com/your-repo/actions
# The enhanced-ci-cd.yml workflow will automatically run on commits
```

## 🌐 Live Deployment Options

### Option 1: Railway (Fastest)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Add Procfile
echo "web: uvicorn app.advanced_api:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
railway deploy
```

### Option 2: Render (Free Tier)
```bash
# Create render.yaml
cat > render.yaml << 'EOF'
services:
  - type: web
    name: churn-analysis-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.advanced_api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
EOF
```

### Option 3: Docker Deployment
```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn scipy

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.advanced_api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Build and run
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## 🧪 Testing Your Deployment

### Test Single Prediction
```python
import requests

# Test data
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

# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Test Data Drift Detection
```python
import requests

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
print(response.json())
```

## 📊 Monitoring Dashboard URLs

Once deployed, your system will have these endpoints:

- **API Documentation**: `http://your-domain/docs`
- **Health Check**: `http://your-domain/health`
- **Model Info**: `http://your-domain/model-info`
- **Streamlit App**: https://customerchurnpredictionanalysis.streamlit.app

## 🔧 Environment Variables (Production)

```bash
# Add these to your deployment platform
ENVIRONMENT=production
MODEL_PATH=./models/
API_TITLE="Customer Churn Analysis API"
API_VERSION=1.0.0
```

## ✅ Deployment Checklist

- [ ] FastAPI service running locally
- [ ] All endpoints responding correctly
- [ ] Model predictions working
- [ ] Data drift detection functional
- [ ] CI/CD pipeline configured
- [ ] Production deployment completed
- [ ] Health checks passing
- [ ] Documentation accessible

## 🎯 Success Metrics

Your deployment is successful when:
- ✅ API returns 200 status for health check
- ✅ Predictions return probability scores 0-1
- ✅ Batch predictions process multiple customers
- ✅ Drift detection returns statistical analysis
- ✅ Interactive documentation loads properly

## 🚀 Next Steps

1. **Monitor Performance**: Check API response times and accuracy
2. **Scale Resources**: Adjust server capacity based on usage
3. **Setup Alerts**: Configure monitoring for model drift and errors
4. **Documentation**: Update team on new API endpoints
5. **Testing**: Run comprehensive tests in production environment

## 🎉 Congratulations!

Your Customer Churn Analysis system is now production-ready with:
- 🌐 Advanced FastAPI service
- 📊 Real-time model monitoring  
- 🔄 Automated CI/CD pipeline
- 📈 Data drift detection
- 🛡️ Security and health checks

**🌟 System Status: ENTERPRISE READY ✅**
