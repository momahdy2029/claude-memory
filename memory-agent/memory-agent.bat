@echo off
REM Claude Memory Agent CLI wrapper for Windows
REM Usage: memory-agent start|stop|status|dashboard|install|logs

cd /d "%~dp0"
python memory-agent %*
