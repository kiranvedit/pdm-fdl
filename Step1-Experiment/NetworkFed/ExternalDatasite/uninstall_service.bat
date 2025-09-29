@echo off
REM Service Uninstaller Script
REM This script must be run as Administrator

echo ======================================
echo Factory Datasite Service Uninstaller
echo ======================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click on this file and select "Run as administrator"
    pause
    exit /b 1
)

echo Stopping Factory Datasite Service...
python factory_datasite_service.py stop

echo.
echo Removing Factory Datasite Service...
python factory_datasite_service.py remove

if %errorlevel% equ 0 (
    echo.
    echo ======================================
    echo Service uninstalled successfully!
    echo ======================================
    echo.
    echo The service has been removed from Windows Services.
    echo Log files and configuration files are preserved.
) else (
    echo.
    echo ERROR: Failed to uninstall service!
    echo The service might not be installed or there was an error.
)

echo.
pause
