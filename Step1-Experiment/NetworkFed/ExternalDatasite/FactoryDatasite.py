"""
FactoryDatasite.py (OOP version)
===============================
FEDERATED LEARNING PREDICTIVE MAINTENANCE - FACTORY COMPONENT

CORE OBJECTIVE: 
Implement a privacy-preserving factory datasite that:
1. Receives data distribution from WebOrchestrator (ExperimentOrchestrator)
2. Sets up PySyft datasite server to host private data
3. Receives model architectures from WebOrchestrator
4. Performs local training on private data
5. Returns ONLY trained model parameters (no raw data sharing)

TECHNICAL SPECIFICATIONS:
========================

1. DATA MANAGEMENT:
   - Receive pre-processed data from WebDashboard
   - Host private datasets as PySyft assets
   - NO local data loading or AI4I dataset access

2. PYSYFT DATASITE SERVER:
   - Launch sy.orchestra datasite server with factory identity
   - Create admin user for WebOrchestrator to login
   - Upload received data as private PySyft assets
   - Handle incoming code requests from CentralOrchestrator
   - Execute federated training functions locally

3. FEDERATED LEARNING EXECUTION:
   - Receive model architectures from CentralOrchestrator
   - Perform local training on private factory data
   - Return only model parameters via PySyft
   - NO fallback models - only use received models

4. MODEL HANDLING:
   - NO local model definitions
   - Accept any model architecture from CentralOrchestrator
   - Train received models on local data
   - Return trained model state_dict()

5. PRIVACY & SECURITY:
   - All data remains local and private
   - Only model parameters shared via PySyft
   - Owner approval mechanisms for code execution
   - Secure parameter exchange protocols

Architecture:
- Factory = PySyft Datasite Server (receives data + models)
- WebDashboard = Data distributor (sends preprocessed data)
- CentralOrchestrator = Model provider + aggregator (sends models, receives updates)
"""

import logging
import syft as sy
import torch
import requests
import threading
import time


class FactoryDatasite:

    def __init__(self, factory_name: str, port:int = 8080, hostname: str = "localhost", dashboard_url: str = "http://localhost:5000", dev_mode: bool = False, reset: bool = True, verbose: bool = False):
        import os
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        self.factory_name = factory_name
        self.port = port
        self.hostname = hostname
        self.dashboard_url = dashboard_url
        self.dev_mode = dev_mode
        self.reset = reset
        self.verbose = verbose
        self.heartbeat_time = 30
        self.logger = self.setup_logging()
        self.logger.info(f"🚦 Starting FactoryDatasite ...")
        self.device = self.detect_device()
        self.datasite = None
        self.logger.info(f"FactoryDatasite {self.factory_name} initializing with device: {self.device}")
        self.datasite = self.launch_datasite()
        
        self.admin_client = self.create_admin_user()
        self.register_with_dashboard()
        #self.status_endpoint = self.make_status_endpoint()

        # Start sending heartbeat signals to the dashboard
        threading.Thread(
            target=self.send_heartbeat,
            daemon=True
        ).start()
        
    
    def detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def setup_logging(self):
        logger = logging.getLogger(self.factory_name)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        # Remove all handlers if re-running in the same process
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # File handler
        file_handler = logging.FileHandler(f"logs/{self.factory_name}.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.addHandler(file_handler)
        # Console handler (only if verbose)
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)
        return logger

    def launch_datasite(self):
        self.logger.info("🚀 Launching PySyft datasite for factory...")
        self.datasite = sy.orchestra.launch(
            name=self.factory_name,
            reset=self.reset,
            port=self.port,
            host=self.hostname,
            dev_mode=self.dev_mode,
            association_request_auto_approval=True
        )
        self.logger.info(f"✅ Datasite launched on port {self.datasite.port}")
        return self.datasite
    
    def create_admin_user(self, admin_email: str = "admin@pdm-factory.com", admin_name: str = "admin", admin_password: str = "password"):
        """Create an admin user for the datasite so WebOrchestrator can login."""
        import syft as sy
        from syft.service.user.user import ServiceRole
        self.logger.info(f"👤 Creating admin user '{admin_name}' ...")
            # Login as the default root user
        try:
            admin_client = sy.login(
            url=self.hostname,
            port=self.port,
            email="info@openmined.org",
            password="changethis"
            )
        except Exception as e:
            self.logger.error(f"Failed to login as root user: {e}")
            return
        if not admin_client or not hasattr(admin_client, "account") or not hasattr(admin_client, "users"):
            self.logger.error("admin_client is not properly initialized. Cannot create admin user.")
            return
        if admin_client.account is None or admin_client.users is None:
            self.logger.error("admin_client.account or admin_client.users is None. Cannot create admin user.")
            return
        try:
            # Set password for root user (optional, for security)
            admin_client.account.set_password(admin_password, confirm=False)
            # Create the admin user
            admin_client.users.create(
                name=admin_name,
                email=admin_email,
                password=admin_password,
                role=ServiceRole.ADMIN
                )
            self.logger.info(f"✅ Admin user '{admin_name}' created with password '{admin_password}'.")
            return admin_client
        except Exception as e:
            self.logger.error(f"Failed to create admin user: {e}")

    def register_with_dashboard(self):
        """
        Register this factory datasite with the central dashboard using actual configuration values.
        Note: Registration goes to dashboard (port 8889), heartbeats go to heartbeat manager (port 8888)
        """
        self.logger.info("🔗 Registering factory datasite with central dashboard...")
        registration_data = {
            "factory_name": self.factory_name,
            "factory_hostname": self.hostname,
            "factory_port": self.port,
            "device": self.device,
            "status": "running",
        }
        
        # Determine dashboard URL (registration endpoint - typically port 8889)
        dashboard_registration_url = self.dashboard_url.replace(":8888", ":8889")
        if not ":8889" in dashboard_registration_url:
            # If dashboard_url doesn't have port, add 8889
            from urllib.parse import urlparse
            parsed = urlparse(self.dashboard_url)
            dashboard_registration_url = f"{parsed.scheme}://{parsed.hostname}:8889"
        
        try:
            # Try registering with the dashboard (port 8889)
            response = requests.post(
                f"{dashboard_registration_url}/register_factory",
                json=registration_data,
                timeout=10
            )
            response.raise_for_status()
            self.logger.info(f"✅ Successfully registered with dashboard at {dashboard_registration_url}")
            
        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to register with dashboard at {dashboard_registration_url}: {e}")
            self.logger.info("🔄 Will continue with heartbeat-only mode...")
    
    def make_status_endpoint(self):
        ''' Even though I created it, the status endpoint is not responding in less than 60 seconds.'''
        self.logger.info(f"Creating status API endpoint ...")
        @sy.api_endpoint(
            path="factory.status",
            description="Public status endpoint for factory datasite. Returns name, hostname, port, and health info."
        )
        def status_endpoint(context):
            return {
                "factory_name": self.factory_name,
                "factory_hostname": self.hostname,
                "factory_port": self.port,
                "device": self.device,
                "status": "running",
            }
        try:
            self.admin_client.custom_api.add(endpoint=status_endpoint)
            self.logger.info("✅ Status endpoint added successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to add status endpoint: {e}")

    def send_heartbeat(self):
        """Send enhanced heartbeat to both dashboard and heartbeat manager using actual datasite configuration"""
        while True:
            try:
                # Create enhanced heartbeat payload using actual datasite configuration
                heartbeat_data = {
                    "factory_name": self.factory_name,
                    "hostname": self.hostname,
                    "port": self.port,
                    "device": self.device,
                    "timestamp": time.time()
                }
                
                # Send heartbeat to NetworkFed heartbeat manager (port 8888 - for experiment coordination)
                heartbeat_manager_url = self.dashboard_url
                if ":8889" in heartbeat_manager_url:
                    heartbeat_manager_url = heartbeat_manager_url.replace(":8889", ":8888")
                elif not ":8888" in heartbeat_manager_url:
                    # If no port specified, add 8888 for heartbeat manager
                    from urllib.parse import urlparse
                    parsed = urlparse(self.dashboard_url)
                    heartbeat_manager_url = f"{parsed.scheme}://{parsed.hostname}:8888"
                
                try:
                    manager_response = requests.post(
                        f"{heartbeat_manager_url}/heartbeat",
                        json=heartbeat_data,
                        timeout=3
                    )
                    if manager_response.status_code == 200:
                        print(f"✅ HeartbeatManager heartbeat sent for {self.factory_name}")
                        self.logger.info(f"✅ HeartbeatManager heartbeat sent for {self.factory_name}")
                    else:
                        print(f"⚠️ HeartbeatManager heartbeat failed: {manager_response.status_code}")
                        self.logger.error(f"⚠️ HeartbeatManager heartbeat failed: {manager_response.status_code}")
                except Exception as manager_error:
                    print(f"🔄 HeartbeatManager not available for {self.factory_name}: {manager_error}")
                    self.logger.info(f"🔄 HeartbeatManager not available for {self.factory_name}: {manager_error}")
                    
                # Send heartbeat to dashboard (port 8889 - for UI display)
                dashboard_url = self.dashboard_url
                if ":8888" in dashboard_url:
                    dashboard_url = dashboard_url.replace(":8888", ":8889")
                elif not ":8889" in dashboard_url:
                    # If no port specified, add 8889 for dashboard
                    from urllib.parse import urlparse
                    parsed = urlparse(self.dashboard_url)
                    dashboard_url = f"{parsed.scheme}://{parsed.hostname}:8889"
                
                try:
                    dashboard_response = requests.post(
                        f"{dashboard_url}/heartbeat",
                        json=heartbeat_data,
                        timeout=3
                    )
                    if dashboard_response.status_code == 200:
                        print(f"✅ Dashboard heartbeat sent for {self.factory_name}")
                        self.logger.info(f"✅ HeartbeatManager heartbeat sent for {self.factory_name}")
                    else:
                        print(f"⚠️ Dashboard heartbeat failed: {dashboard_response.status_code}")
                        self.logger.error(f"⚠️ Dashboard heartbeat failed: {dashboard_response.status_code}")
                except Exception as dash_error:
                    print(f"❌ Dashboard heartbeat error: {dash_error}")
                    self.logger.error(f"❌ Dashboard heartbeat error: {dash_error}")
                    
            except Exception as e:
                print(f"💔 Heartbeat error for {self.factory_name}: {e}")
                self.logger.error(f"💔 Heartbeat error for {self.factory_name}: {e}")
                
            time.sleep(self.heartbeat_time)  # Send heartbeat every N seconds



       

#
# Main entry point to run the datasite continuously
def main():
    """
    Main function to launch and keep the PySyft datasite running.
    The datasite will run until interrupted by Ctrl+C.
    """
    import argparse
    import time
    import signal
    import sys
    parser = argparse.ArgumentParser(description="Run FactoryDatasite server.")
    parser.add_argument("--factory_name", type=str, default="pdm_factory_datasite", help="Name of the factory datasite.")
    parser.add_argument("--hostname", type=str, default="localhost", help="Hostname to bind the datasite.")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the datasite on.")
    parser.add_argument("--dashboard_url", type=str, default="http://localhost:5000", help="URL of the WebOrchestrator dashboard.")
    parser.add_argument("--dev_mode", action="store_true", help="Enable development mode for the datasite.")
    parser.add_argument("--reset", action="store_true", help="Reset the datasite state on startup.")
    parser.add_argument("--verbose", action="store_true", help="Print logs to console as well as file.")
    args = parser.parse_args()
    factory = FactoryDatasite(
        factory_name=args.factory_name,
        port=args.port,
        hostname=args.hostname,
        dashboard_url=args.dashboard_url,
        dev_mode=args.dev_mode,
        reset=args.reset,
        verbose=args.verbose
    )
    if args.verbose:
        print("\n[INFO] PySyft datasite is running. Press Ctrl+C to stop.")

    # Ignore all signals except SIGINT (Ctrl+C)
    def handler(signum, frame):
        print(f"[INFO] Ignoring signal: {signum}")
    for sig in [signal.SIGTERM, signal.SIGBREAK, signal.SIGABRT]:
        try:
            signal.signal(sig, handler)
        except (AttributeError, ValueError):
            pass  # Not all signals are available on all platforms
    try:
        while True:
            time.sleep(1)  # Keeps the process alive, only exits on Ctrl+C
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down the PySyft datasite.")


if __name__ == "__main__":
    main()  