@echo off
cd /d "%~dp0"

echo Stopping any existing server...
taskkill /f /im uvicorn.exe >nul 2>&1

echo Starting WRLD Sync...
start "" http://localhost:8000
.venv\Scripts\uvicorn.exe app:app

pause
