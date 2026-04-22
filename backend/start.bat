@echo off
echo [ML Agent Backend] Starting server...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    D:\python.exe -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies (use python -m pip since pip may not be on PATH)
echo Installing dependencies...
python -m pip install -r requirements.txt

REM Start server
echo.
echo Server starting at http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
