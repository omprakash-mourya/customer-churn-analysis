@echo off
echo 🔥 Starting Customer Churn Fire Dashboard...
echo.
echo Using virtual environment: %CD%\venv
echo.

REM Activate virtual environment and run Streamlit
call venv\Scripts\activate.bat
streamlit run app/streamlit_app.py

echo.
echo Dashboard stopped.
pause
