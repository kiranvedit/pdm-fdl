"""
Factory Datasite Windows Service
================================
Windows service wrapper for FactoryDatasite.py to run as a background service.

Installation:
1. Install pywin32: pip install pywin32
2. Run as administrator: python factory_datasite_service.py install
3. Start service: python factory_datasite_service.py start
   OR: net start FactoryDatasiteService
   OR: Use Windows Services manager

Removal:
1. Stop service: python factory_datasite_service.py stop
2. Remove service: python factory_datasite_service.py remove

Service Management:
- Start: python factory_datasite_service.py start
- Stop: python factory_datasite_service.py stop
- Restart: python factory_datasite_service.py restart
- Status: python factory_datasite_service.py status
"""

import sys
import os
import time
import json
import logging
import threading
import subprocess
from pathlib import Path

try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
except ImportError:
    print("Error: pywin32 is required. Install with: pip install pywin32")
    sys.exit(1)


class FactoryDatasiteService(win32serviceutil.ServiceFramework):
    """Windows service wrapper for Factory Datasite"""
    
    # Service configuration
    _svc_name_ = "FactoryDatasiteService"
    _svc_display_name_ = "Factory Datasite Service"
    _svc_description_ = "PySyft Factory Datasite for Federated Learning PDM System"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = False
        self.factory_process = None
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
        self.logger.info("Factory Datasite Service initialized")
    
    def setup_logging(self):
        """Setup logging for the service"""
        # Create logs directory
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "factory_datasite_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FactoryDatasiteService")
    
    def load_config(self):
        """Load service configuration from JSON file"""
        config_file = Path(__file__).parent / "service_config.json"
        
        # Default configuration
        default_config = {
            "factory_name": "factory_03",
            "port": 8083,
            "hostname": "localhost",
            "dashboard_url": "http://localhost:8888",
            "dev_mode": True,
            "reset": True,
            "verbose": True,
            "python_executable": sys.executable,
            "script_path": str(Path(__file__).parent / "FactoryDatasite.py")
        }
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                self.logger.info(f"Loaded configuration from {config_file}")
            else:
                config = default_config
                # Save default config
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                self.logger.info(f"Created default configuration at {config_file}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def build_command(self):
        """Build the command to start FactoryDatasite.py"""
        cmd = [
            self.config["python_executable"],
            self.config["script_path"],
            "--factory_name", self.config["factory_name"],
            "--port", str(self.config["port"]),
            "--hostname", self.config["hostname"],
            "--dashboard_url", self.config["dashboard_url"]
        ]
        
        if self.config.get("dev_mode", False):
            cmd.append("--dev_mode")
        
        if self.config.get("reset", False):
            cmd.append("--reset")
        
        if self.config.get("verbose", False):
            cmd.append("--verbose")
        
        return cmd
    
    def SvcStop(self):
        """Called when the service is asked to stop"""
        self.logger.info("Service stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.running = False
        
        # Terminate the factory process
        if self.factory_process and self.factory_process.poll() is None:
            try:
                self.logger.info("Terminating factory datasite process")
                self.factory_process.terminate()
                # Wait up to 10 seconds for graceful shutdown
                try:
                    self.factory_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process didn't terminate gracefully, killing it")
                    self.factory_process.kill()
                self.logger.info("Factory datasite process terminated")
            except Exception as e:
                self.logger.error(f"Error terminating process: {e}")
        
        win32event.SetEvent(self.hWaitStop)
        self.logger.info("Service stopped")
    
    def SvcDoRun(self):
        """Called when the service is asked to start"""
        self.logger.info("Service starting")
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        self.running = True
        self.main()
    
    def main(self):
        """Main service loop"""
        self.logger.info("Factory Datasite Service started")
        
        while self.running:
            try:
                # Build command
                cmd = self.build_command()
                self.logger.info(f"Starting factory datasite with command: {' '.join(cmd)}")
                
                # Start the factory datasite process
                self.factory_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(Path(__file__).parent)
                )
                
                self.logger.info(f"Factory datasite process started with PID: {self.factory_process.pid}")
                
                # Monitor the process
                while self.running and self.factory_process.poll() is None:
                    # Check if service should stop
                    if win32event.WaitForSingleObject(self.hWaitStop, 1000) == win32event.WAIT_OBJECT_0:
                        break
                
                # If we exit the loop and the service is still running, the process crashed
                if self.running and self.factory_process.poll() is not None:
                    return_code = self.factory_process.returncode
                    stdout, stderr = self.factory_process.communicate()
                    
                    self.logger.error(f"Factory datasite process exited with code {return_code}")
                    if stdout:
                        self.logger.error(f"STDOUT: {stdout}")
                    if stderr:
                        self.logger.error(f"STDERR: {stderr}")
                    
                    # Wait before restarting
                    if self.running:
                        self.logger.info("Restarting factory datasite in 5 seconds...")
                        time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                if self.running:
                    self.logger.info("Restarting in 10 seconds...")
                    time.sleep(10)
        
        self.logger.info("Service main loop ended")


def install_service():
    """Install the service"""
    try:
        win32serviceutil.InstallService(
            FactoryDatasiteService._svc_reg_class_,
            FactoryDatasiteService._svc_name_,
            FactoryDatasiteService._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START,
            description=FactoryDatasiteService._svc_description_
        )
        print(f"Service '{FactoryDatasiteService._svc_display_name_}' installed successfully")
        print("Start the service with: net start FactoryDatasiteService")
    except Exception as e:
        print(f"Error installing service: {e}")


def remove_service():
    """Remove the service"""
    try:
        win32serviceutil.RemoveService(FactoryDatasiteService._svc_name_)
        print(f"Service '{FactoryDatasiteService._svc_display_name_}' removed successfully")
    except Exception as e:
        print(f"Error removing service: {e}")


def start_service():
    """Start the service"""
    try:
        win32serviceutil.StartService(FactoryDatasiteService._svc_name_)
        print(f"Service '{FactoryDatasiteService._svc_display_name_}' started successfully")
    except Exception as e:
        print(f"Error starting service: {e}")


def stop_service():
    """Stop the service"""
    try:
        win32serviceutil.StopService(FactoryDatasiteService._svc_name_)
        print(f"Service '{FactoryDatasiteService._svc_display_name_}' stopped successfully")
    except Exception as e:
        print(f"Error stopping service: {e}")


def restart_service():
    """Restart the service"""
    try:
        stop_service()
        time.sleep(2)
        start_service()
    except Exception as e:
        print(f"Error restarting service: {e}")


def service_status():
    """Check service status"""
    try:
        status = win32serviceutil.QueryServiceStatus(FactoryDatasiteService._svc_name_)
        status_map = {
            win32service.SERVICE_STOPPED: "STOPPED",
            win32service.SERVICE_START_PENDING: "START_PENDING",
            win32service.SERVICE_STOP_PENDING: "STOP_PENDING",
            win32service.SERVICE_RUNNING: "RUNNING",
            win32service.SERVICE_CONTINUE_PENDING: "CONTINUE_PENDING",
            win32service.SERVICE_PAUSE_PENDING: "PAUSE_PENDING",
            win32service.SERVICE_PAUSED: "PAUSED"
        }
        current_status = status_map.get(status[1], "UNKNOWN")
        print(f"Service '{FactoryDatasiteService._svc_display_name_}' status: {current_status}")
    except Exception as e:
        print(f"Error checking service status: {e}")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Called by Windows Service Manager
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(FactoryDatasiteService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Called from command line
        command = sys.argv[1].lower()
        
        if command == 'install':
            install_service()
        elif command == 'remove':
            remove_service()
        elif command == 'start':
            start_service()
        elif command == 'stop':
            stop_service()
        elif command == 'restart':
            restart_service()
        elif command == 'status':
            service_status()
        else:
            print("Usage:")
            print("  python factory_datasite_service.py install   - Install the service")
            print("  python factory_datasite_service.py remove    - Remove the service")
            print("  python factory_datasite_service.py start     - Start the service")
            print("  python factory_datasite_service.py stop      - Stop the service")
            print("  python factory_datasite_service.py restart   - Restart the service")
            print("  python factory_datasite_service.py status    - Check service status")
