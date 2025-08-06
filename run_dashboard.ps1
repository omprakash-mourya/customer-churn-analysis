# Customer Churn Fire Dashboard Launcher
Write-Host "🔥 Starting Customer Churn Fire Dashboard..." -ForegroundColor Yellow
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    
    # Activate virtual environment
    Write-Host "🚀 Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
    
    # Check if Streamlit is installed
    $streamlitCheck = & python -c "import streamlit; print('OK')" 2>$null
    if ($streamlitCheck -eq "OK") {
        Write-Host "✅ Streamlit is available" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Starting dashboard at: http://localhost:8501" -ForegroundColor Yellow
        Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor White
        Write-Host ""
        
        # Run Streamlit app
        streamlit run app/streamlit_app.py
    } else {
        Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Red
        pip install streamlit==1.29.0
        Write-Host "🚀 Retrying..." -ForegroundColor Cyan
        streamlit run app/streamlit_app.py
    }
} else {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor White
    Write-Host "Then run: .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "And install requirements: pip install -r requirements.txt" -ForegroundColor White
}

Write-Host ""
Write-Host "Dashboard stopped." -ForegroundColor Yellow
