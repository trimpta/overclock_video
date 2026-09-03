@echo off
set "VENV_NAME=.venv"
set "FORCE_INSTALL=0"

if /I "%~1"=="--install" set "FORCE_INSTALL=1"
if /I "%~1"=="install" set "FORCE_INSTALL=1"

:: 1. Create venv if missing
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure python is installed and in PATH.
        pause
        exit /b 1
    )
    goto :install_and_launch
)

:: 2. Existing venv: activate; install only if forced
echo Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"

if "%FORCE_INSTALL%"=="1" goto :install_deps
goto :launch

:install_and_launch
echo Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"

:install_deps
echo Upgrading pip...
python -m pip install --upgrade pip

if exist "requirements.txt" (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo requirements.txt not found, skipping dependency installation.
)

:launch
echo Setup complete! Launching Overclock Video...
python overclock.py

pause
