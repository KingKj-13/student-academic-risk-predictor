@echo off
rem Windows: creates a virtual environment, trains the model once and starts the app.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv || python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt || goto :error
if not exist models\risk_model.joblib python train.py || goto :error
echo.
echo Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.
python -m uvicorn main:app --host 127.0.0.1 --port 8000
goto :eof

:error
echo Setup failed. Read the message above.
pause
