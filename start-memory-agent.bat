@echo off
echo ============================================
echo Claude Memory System - Starting...
echo ============================================
echo.

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not running - starting it...
    start "" ollama serve
    timeout /t 3 > nul
    curl -s http://localhost:11434/api/tags > nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to start Ollama. Please start it manually with: ollama serve
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Ollama is running

REM Check if memory agent is already running
curl -s http://localhost:8102/health > nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Memory Agent already running on port 8102
    echo.
    echo Opening dashboard...
    start "" http://localhost:8102/dashboard
    exit /b 0
)

REM Activate virtual environment and start the agent
cd /d "C:\Users\moham\Desktop\Claude Memory\memory-agent"
call venv\Scripts\activate
echo [OK] Virtual environment activated
echo.
echo Starting Memory Agent on http://localhost:8102...
echo Dashboard will open automatically...
echo.

REM Start agent in background and open dashboard
set PORT=8102
start "" python main.py

REM Wait for agent to start
timeout /t 3 > nul

REM Open dashboard
start "" http://localhost:8102/dashboard

echo.
echo ============================================
echo Memory System Started!
echo - Memory Agent: http://localhost:8102
echo - Dashboard: http://localhost:8102/dashboard
echo - Ollama: http://localhost:11434
echo ============================================
echo.
echo You can close this window. The agent runs in the background.
echo To stop: Close the Python window or use Task Manager.
pause
