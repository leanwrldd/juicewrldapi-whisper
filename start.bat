@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Python was not found on your PATH.
    echo   Install Python 3.10+ from https://python.org/downloads/
    echo   and make sure to check "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

python launch.py %*

echo.
pause
