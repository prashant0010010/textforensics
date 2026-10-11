@echo off
REM TextForensics Local - Windows launcher. Creates a virtual environment on first run, installs the CORE dependencies,
REM and starts the app. Internet is needed only for the one-time "pip install" step.
setlocal
cd /d "%~dp0"
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=false

where py >nul 2>nul
if %ERRORLEVEL%==0 ( set "PY=py -3" ) else ( set "PY=python" )

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 ( echo Could not create the virtual environment. Install Python 3.10-3.12 from python.org and tick "Add to PATH". & pause & exit /b 1 )
    echo Installing core requirements ^(one-time, needs internet^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 ( echo Dependency installation failed. & pause & exit /b 1 )
)

echo Starting TextForensics Local at http://localhost:8501  ^(all processing stays on this computer^)
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
pause
