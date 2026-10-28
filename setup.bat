@echo off
REM setup.bat - Setup & Startup Script for AI Interview & Coding Copilot (Windows 10/11)
setlocal enabledelayedexpansion
cd /d "%~dp0"

set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_PARALLEL=1

set MODEL_NAME=%1
if "%MODEL_NAME%"=="" set MODEL_NAME=qwen2.5-coder:3b
set OLLAMA_HOST_URL=http://localhost:11434

echo ============================================================
echo  AI Interview ^& Coding Copilot Setup (Windows 10/11)
echo  OLLAMA_MAX_LOADED_MODELS=%OLLAMA_MAX_LOADED_MODELS%
echo  OLLAMA_NUM_PARALLEL=%OLLAMA_NUM_PARALLEL%
echo ============================================================

REM 1. Locate working Python 3 interpreter (prefer py -3 launcher on Windows, then python)
set PY_CMD=python
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3 --version >nul 2>nul
    if !ERRORLEVEL! equ 0 set PY_CMD=py -3
)

REM 2. Set up Python virtual environment & install dependencies
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment (.venv) using %PY_CMD%...
    %PY_CMD% -m venv .venv
)

if exist "requirements.txt" (
    echo [INFO] Installing dependencies from requirements.txt...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

REM 3. Create .env template if not present
if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo [INFO] Created .env from .env.example (Cloud mode active by default).
    )
)

REM 4. Optional local Ollama setup (if ollama is installed)
where ollama >nul 2>nul
if %ERRORLEVEL% equ 0 (
    curl -fsS "%OLLAMA_HOST_URL%/api/tags" >nul 2>nul
    if !ERRORLEVEL! neq 0 (
        echo [INFO] Starting local Ollama server...
        start /B ollama serve >nul 2>nul
        timeout /t 3 /nobreak >nul
    )
    echo [INFO] Pulling local coding model '%MODEL_NAME%'...
    ollama pull %MODEL_NAME%
) else (
    echo [INFO] Using Gemini Cloud mode (0 MB local LLM RAM).
)

echo.
echo [SUCCESS] Setup complete!
echo To launch on Windows, run:
echo   run.bat
endlocal
