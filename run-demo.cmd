@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py
exit /b %errorlevel%
:failed
echo Setup failed. Install Python 3.11 or 3.12 and check the error above.
pause
exit /b 1
