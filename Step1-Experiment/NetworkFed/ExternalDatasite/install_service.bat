@echo off
REM Quick Service Installation Script
REM This script must be run as Administrator

echo ======================================
echo Factory Datasite Service Installer
echo ======================================
echo.
echo This script will:
echo 1. Install pywin32 (if not already installed)
echo 2. Install the Factory Datasite Windows Service
echo 3. Configure the service to start automatically
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click on this file and select "Run as administrator"
    pause
    exit /b 1
)

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and make sure it's in your system PATH.
    pause
    exit /b 1
)

echo Python found. Installing required dependencies...
echo.

echo Installing pywin32...
pip install pywin32
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pywin32!
    pause
    exit /b 1
)

echo.
echo Installing Factory Datasite Service...
python factory_datasite_service.py install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install service!
    pause
    exit /b 1
)

echo.
echo ======================================
echo Installation completed successfully!
echo ======================================
echo.
echo You can now:
echo 1. Start the service: service_manager.bat start
echo 2. Check status: service_manager.bat status
echo 3. View logs: service_manager.bat logs
echo.
echo The service is configured to start automatically on system boot.
echo.
echo Configuration file: service_config.json
echo Edit this file to change factory settings.
echo.
pause
