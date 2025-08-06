# Development Environment Setup Script for Windows
# Run this script in PowerShell to set up the development environment

Write-Host "🔥 Customer Churn Fire Project - Development Setup" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Yellow

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python is not installed. Please install Python 3.9+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "🐍 Python version: $pythonVersion" -ForegroundColor Green

# Check if Git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is not installed. Please install Git from https://git-scm.com" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate virtual environment
Write-Host "🚀 Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "📚 Installing Python packages..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install development dependencies
Write-Host "🔧 Installing development dependencies..." -ForegroundColor Cyan
pip install pytest pytest-cov pytest-xdist black flake8 mypy bandit safety pre-commit

# Create data directory
Write-Host "📁 Creating data directory..." -ForegroundColor Cyan
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

# Create models directory
Write-Host "🤖 Creating models directory..." -ForegroundColor Cyan
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models"
}

# Create logs directory
Write-Host "📝 Creating logs directory..." -ForegroundColor Cyan
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
}

# Set up pre-commit hooks (optional)
Write-Host "🔗 Setting up pre-commit hooks..." -ForegroundColor Cyan
try {
    pre-commit install
    Write-Host "✅ Pre-commit hooks installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Pre-commit hooks setup failed, but that's okay for development" -ForegroundColor Yellow
}

# Run initial tests
Write-Host "🧪 Running initial tests..." -ForegroundColor Cyan
python -m pytest tests/ -v

# Display setup completion
Write-Host "" -ForegroundColor White
Write-Host "🎉 Development environment setup completed!" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   2. Run setup: python main.py setup" -ForegroundColor White
Write-Host "   3. Train models: python main.py train" -ForegroundColor White
Write-Host "   4. Start API: uvicorn app.api:app --reload" -ForegroundColor White
Write-Host "   5. Start Dashboard: streamlit run app/streamlit_app.py" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "📚 Available Commands:" -ForegroundColor Yellow
Write-Host "   • python main.py --help" -ForegroundColor White
Write-Host "   • pytest tests/" -ForegroundColor White
Write-Host "   • black . (code formatting)" -ForegroundColor White
Write-Host "   • flake8 . (linting)" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "🔗 Useful URLs (after starting services):" -ForegroundColor Yellow
Write-Host "   • API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "   • Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Happy coding! 🚀" -ForegroundColor Green
