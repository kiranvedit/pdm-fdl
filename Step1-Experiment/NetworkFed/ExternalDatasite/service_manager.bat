@echo off
REM Factory Datasite Service Management Script
REM Run as Administrator for install/remove operations

setlocal
set SERVICE_NAME=FactoryDatasiteService
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%factory_datasite_service.py

echo ================================
echo Factory Datasite Service Manager
echo ================================
echo.

if "%1"=="install" goto install
if "%1"=="remove" goto remove
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="status" goto status
if "%1"=="logs" goto logs

:menu
echo Available commands:
echo   install  - Install the Windows service (requires admin)
echo   remove   - Remove the Windows service (requires admin)
echo   start    - Start the service
echo   stop     - Stop the service
echo   restart  - Restart the service
echo   status   - Check service status
echo   logs     - View service logs
echo.
echo Usage: %0 [command]
echo Example: %0 install
goto end

:install
echo Installing Factory Datasite Service...
python "%PYTHON_SCRIPT%" install
if %errorlevel% equ 0 (
    echo.
    echo Service installed successfully!
    echo You can now start it with: %0 start
    echo Or use Windows Services manager
) else (
    echo.
    echo Installation failed! Make sure you're running as Administrator.
)
goto end

:remove
echo Stopping service first...
python "%PYTHON_SCRIPT%" stop
echo Removing Factory Datasite Service...
python "%PYTHON_SCRIPT%" remove
if %errorlevel% equ 0 (
    echo Service removed successfully!
) else (
    echo Removal failed! Make sure you're running as Administrator.
)
goto end

:start
echo Starting Factory Datasite Service...
python "%PYTHON_SCRIPT%" start
if %errorlevel% equ 0 (
    echo Service started successfully!
    echo Check status with: %0 status
) else (
    echo Failed to start service. Check logs with: %0 logs
)
goto end

:stop
echo Stopping Factory Datasite Service...
python "%PYTHON_SCRIPT%" stop
if %errorlevel% equ 0 (
    echo Service stopped successfully!
) else (
    echo Failed to stop service.
)
goto end

:restart
echo Restarting Factory Datasite Service...
python "%PYTHON_SCRIPT%" restart
if %errorlevel% equ 0 (
    echo Service restarted successfully!
) else (
    echo Failed to restart service. Check logs with: %0 logs
)
goto end

:status
echo Checking Factory Datasite Service status...
python "%PYTHON_SCRIPT%" status
echo.
echo Windows Service status:
sc query %SERVICE_NAME%
goto end

:logs
echo Opening service logs...
if exist "%SCRIPT_DIR%logs\factory_datasite_service.log" (
    type "%SCRIPT_DIR%logs\factory_datasite_service.log"
) else (
    echo No service logs found at %SCRIPT_DIR%logs\factory_datasite_service.log
)
echo.
if exist "%SCRIPT_DIR%logs\factory_03.log" (
    echo Factory datasite logs:
    type "%SCRIPT_DIR%logs\factory_03.log"
) else (
    echo No factory datasite logs found at %SCRIPT_DIR%logs\factory_03.log
)
goto end

:end
echo.
pause
