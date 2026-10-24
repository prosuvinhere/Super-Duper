@echo off
REM run.bat - Launch AI Interview & Coding Copilot on Windows
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe main.py %*
) else (
    python main.py %*
)
