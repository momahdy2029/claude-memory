@echo off
:: Claude Memory Manager Launcher
:: Double-click to launch the Memory Manager GUI

cd /d "%~dp0memory-agent"

:: Use pythonw.exe to run without console window
if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "manager.py"
) else (
    :: Fallback to python.exe if pythonw doesn't exist
    start "" "venv\Scripts\python.exe" "manager.py"
)
