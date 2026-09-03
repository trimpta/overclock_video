@echo off
set "VENV_NAME=.venv"

:: 1. Check if the venv exists; if not, create it
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure python is installed and in PATH.
        pause
        exit /b 1
    )
)

:: 2. Activate the virtual environment
echo Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"

:: 3. Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install requirements if the file exists
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

echo Setup complete! Launching Overclock Video...
python overclock.py

pause
