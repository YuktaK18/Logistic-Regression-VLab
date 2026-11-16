@echo off
title Logistic Regression VLab
cd /d "%~dp0"
echo ==================================================
echo   Logistic Regression VLab  (Streamlit)
echo ==================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found. Install Python 3.10-3.13 from python.org
    echo         and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment ^(first run only^)...
    python -m venv venv
    if errorlevel 1 ( echo [ERROR] Could not create venv. & pause & exit /b 1 )
)
call venv\Scripts\activate.bat

if not exist venv\.installed (
    echo Installing packages ^(first run only, needs internet^)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 ( echo [ERROR] Package installation failed. & pause & exit /b 1 )
    echo done> venv\.installed
)

rem Stop Streamlit asking for an e-mail address on first run
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

python check_env.py
if errorlevel 2 goto backup
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo Starting the Streamlit lab - your browser will open at http://localhost:8501
echo Close this window or press Ctrl+C to stop.
python -m streamlit run app.py
goto end

:backup
echo.
echo ---- Streamlit/pandas is blocked on this PC. Starting the BACKUP version instead ----
python flask_backup\app_flask.py

:end
pause
