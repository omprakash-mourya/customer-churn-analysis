#!/bin/bash

# Customer Churn Fire Project Deployment Script
# This script handles deployment to various platforms

set -e  # Exit on any error

echo "🔥 Customer Churn Fire Project Deployment Script"
echo "================================================"

# Configuration
PROJECT_NAME="customer-churn-fire"
DOCKER_IMAGE="churn-prediction:latest"
STREAMLIT_PORT=8501
FASTAPI_PORT=8000

# Parse command line arguments
PLATFORM=${1:-"local"}
SERVICE=${2:-"all"}

case $PLATFORM in
    "local")
        echo "🏠 Deploying locally..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is not installed. Please install Docker first."
            exit 1
        fi
        
        # Build Docker image
        echo "🔨 Building Docker image..."
        docker build -t $DOCKER_IMAGE .
        
        if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "api" ]; then
            echo "🔌 Starting FastAPI service..."
            docker run -d --name churn-api \
                -p $FASTAPI_PORT:8000 \
                -v $(pwd)/models:/app/models \
                -v $(pwd)/data:/app/data \
                $DOCKER_IMAGE uvicorn app.api:app --host 0.0.0.0 --port 8000
            echo "✅ FastAPI running at http://localhost:$FASTAPI_PORT"
        fi
        
        if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "dashboard" ]; then
            echo "🎨 Starting Streamlit dashboard..."
            docker run -d --name churn-dashboard \
                -p $STREAMLIT_PORT:8501 \
                -v $(pwd)/models:/app/models \
                -v $(pwd)/data:/app/data \
                $DOCKER_IMAGE streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
            echo "✅ Streamlit dashboard running at http://localhost:$STREAMLIT_PORT"
        fi
        ;;
        
    "render")
        echo "☁️ Deploying to Render..."
        
        # Create render.yaml if it doesn't exist
        if [ ! -f "render.yaml" ]; then
            echo "📝 Creating render.yaml configuration..."
            cat > render.yaml << EOF
services:
  - type: web
    name: churn-prediction-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.api:app --host 0.0.0.0 --port \$PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.16
        
  - type: web
    name: churn-prediction-dashboard
    env: python  
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/streamlit_app.py --server.port \$PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.16
EOF
        fi
        
        echo "✅ Render configuration ready!"
        echo "📋 Next steps:"
        echo "   1. Push your code to GitHub"
        echo "   2. Connect your GitHub repo to Render"
        echo "   3. Deploy using the render.yaml configuration"
        ;;
        
    "heroku")
        echo "☁️ Deploying to Heroku..."
        
        # Check if Heroku CLI is installed
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI is not installed. Please install it first."
            exit 1
        fi
        
        # Create Procfile if it doesn't exist
        if [ ! -f "Procfile" ]; then
            echo "📝 Creating Procfile..."
            cat > Procfile << EOF
web: uvicorn app.api:app --host 0.0.0.0 --port \$PORT
streamlit: streamlit run app/streamlit_app.py --server.port \$PORT --server.address 0.0.0.0
EOF
        fi
        
        # Create runtime.txt
        echo "python-3.9.16" > runtime.txt
        
        echo "🚀 Creating Heroku apps..."
        
        # Create API app
        heroku create $PROJECT_NAME-api || echo "App might already exist"
        heroku config:set BUILDPACK_URL=https://github.com/heroku/heroku-buildpack-python.git --app $PROJECT_NAME-api
        
        # Create Dashboard app  
        heroku create $PROJECT_NAME-dashboard || echo "App might already exist"
        heroku config:set BUILDPACK_URL=https://github.com/heroku/heroku-buildpack-python.git --app $PROJECT_NAME-dashboard
        
        echo "✅ Heroku apps created!"
        echo "📋 Next steps:"
        echo "   1. git push heroku main"
        echo "   2. Configure process types in Heroku dashboard"
        ;;
        
    "aws")
        echo "☁️ AWS deployment not implemented yet."
        echo "📋 Recommended approach:"
        echo "   • Use AWS ECS with the Docker image"
        echo "   • Deploy via AWS App Runner"
        echo "   • Use AWS Lambda for serverless deployment"
        ;;
        
    "azure")
        echo "☁️ Azure deployment not implemented yet." 
        echo "📋 Recommended approach:"
        echo "   • Use Azure Container Instances"
        echo "   • Deploy via Azure App Service"
        echo "   • Use Azure Functions for serverless deployment"
        ;;
        
    *)
        echo "❌ Unknown platform: $PLATFORM"
        echo "📋 Supported platforms:"
        echo "   • local    - Deploy locally with Docker"
        echo "   • render   - Deploy to Render.com"
        echo "   • heroku   - Deploy to Heroku"
        echo "   • aws      - AWS deployment (coming soon)"
        echo "   • azure    - Azure deployment (coming soon)"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment script completed!"
echo "🔗 Quick Links:"
echo "   📊 Dashboard: http://localhost:$STREAMLIT_PORT"
echo "   🔌 API: http://localhost:$FASTAPI_PORT"
echo "   📖 API Docs: http://localhost:$FASTAPI_PORT/docs"
echo ""
echo "🛠️ Management Commands:"
echo "   • Stop services: docker stop churn-api churn-dashboard"
echo "   • Remove containers: docker rm churn-api churn-dashboard"  
echo "   • View logs: docker logs churn-api"
