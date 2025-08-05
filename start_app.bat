@echo off
echo 🎯 Starting SaaS Lead Scorer Full Stack Application
echo =========================================================
echo.

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ Error: app.py not found!
    echo Please run this script from the saasquatchleads directory.
    pause
    exit /b 1
)

echo ✅ Found app.py - starting services...
echo.

echo 🚀 Starting FastAPI backend on port 8501...
start /b python -m uvicorn app:app --host 0.0.0.0 --port 8501 --reload

echo ⏳ Waiting for FastAPI to initialize...
timeout /t 5 >nul

echo 🎨 Starting Streamlit frontend on port 8502...
echo.
echo 📊 Dashboard will open at: http://localhost:8502
echo 🔗 API documentation at: http://localhost:8501/docs
echo.

REM Start Streamlit (this will block)
python -m streamlit run dashboard.py --server.port=8502 --server.address=0.0.0.0 --browser.gatherUsageStats=false
