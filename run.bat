@echo off
REM run.bat - Launch AI Interview & Coding Copilot on Windows 10/11
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Running setup.bat first...
    call setup.bat
)
.venv\Scripts\python.exe -c "import PyQt6, mss, rapidocr_onnxruntime" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing missing Python dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
.venv\Scripts\python.exe main.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Copilot exited with code %ERRORLEVEL%.
    pause
)
