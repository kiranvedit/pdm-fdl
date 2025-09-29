"""
Heartbeat Manager
================
Receives and tracks heartbeat signals from factory datasites.
Provides datasite availability status for experiment orchestration.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from flask import Flask, request, jsonify
import logging


class HeartbeatManager:
    """
    Manages heartbeat signals from factory datasites.
    Provides real-time availability status for experiment coordination.
    """
    
    def __init__(self, port: int = 8888, heartbeat_timeout: int = 90):
        """
        Initialize heartbeat manager.
        
        Args:
            port: Port to listen for heartbeat signals
            heartbeat_timeout: Seconds after which datasite is considered offline
        """
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        
        # Datasite status tracking
        self.datasite_status: Dict[str, Dict] = {}
        self.status_lock = threading.Lock()
        
        # Flask app for receiving heartbeats
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging
        
        # Setup routes
        self._setup_routes()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup Flask routes for heartbeat reception."""
        
        @self.app.route('/heartbeat', methods=['POST'])
        def receive_heartbeat():
            """Receive heartbeat signal from factory datasite."""
            try:
                data = request.get_json()
                if not data:
                    self.logger.warning("Received heartbeat with no JSON data")
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                factory_name = data.get('factory_name')
                
                if not factory_name:
                    self.logger.warning("Received heartbeat missing factory_name")
                    return jsonify({'error': 'Missing factory_name'}), 400
                
                # Extract client IP for hostname tracking
                client_ip = request.remote_addr
                
                # Update datasite status with extended information and thread safety
                with self.status_lock:
                    existing = self.datasite_status.get(factory_name, {})
                    self.datasite_status[factory_name] = {
                        'last_heartbeat': datetime.now(),
                        'status': 'online',
                        'heartbeat_count': existing.get('heartbeat_count', 0) + 1,
                        'hostname': data.get('hostname', client_ip),
                        'port': data.get('port', 8080),
                        'device': data.get('device', 'unknown'),
                        'client_ip': client_ip,
                        'last_update': datetime.now().isoformat()
                    }
                
                # Log successful heartbeat for debugging
                heartbeat_count = self.datasite_status[factory_name]['heartbeat_count']
                self.logger.debug(f"✅ Heartbeat received from {factory_name} (#{heartbeat_count}) from {client_ip}")
                
                return jsonify({'status': 'ok', 'received_at': datetime.now().isoformat()}), 200
                
            except Exception as e:
                self.logger.error(f"Error processing heartbeat: {e}")
                return jsonify({'error': f'Internal error: {str(e)}'}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get current status of all datasites."""
            return jsonify(self.get_all_datasite_status())
    
    def start(self):
        """Start heartbeat manager in background thread."""
        if self.running:
            self.logger.warning("Heartbeat manager already running")
            return
        
        self.running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_offline_datasites, daemon=True)
        self.cleanup_thread.start()
        
        # Start Flask app in background thread with improved error handling
        flask_thread = threading.Thread(
            target=self._run_flask_app,
            daemon=True
        )
        flask_thread.start()
        
        self.logger.info(f"Heartbeat manager starting on port {self.port} with {self.heartbeat_timeout}s timeout")
        time.sleep(2)  # Give Flask more time to start
        
        # Verify Flask is running
        try:
            import requests
            test_response = requests.get(f"http://localhost:{self.port}/status", timeout=2)
            if test_response.status_code == 200:
                self.logger.info(f"✅ Heartbeat manager successfully started and responding on port {self.port}")
            else:
                self.logger.warning(f"⚠️ Heartbeat manager started but not responding correctly")
        except Exception as e:
            self.logger.error(f"❌ Heartbeat manager startup verification failed: {e}")
    
    def _run_flask_app(self):
        """Run Flask app with improved error handling."""
        try:
            # Disable Flask request logging to reduce noise
            import logging as flask_logging
            flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)
            
            self.app.run(
                host='0.0.0.0', 
                port=self.port, 
                debug=False, 
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Flask app failed to start: {e}")
            self.running = False
    
    def stop(self):
        """Stop heartbeat manager."""
        self.running = False
        self.logger.info("Heartbeat manager stopped")
    
    def _cleanup_offline_datasites(self):
        """Background thread to mark offline datasites and sync with dashboard."""
        self.logger.info(f"Cleanup thread started with {self.heartbeat_timeout}s timeout threshold")
        
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = timedelta(seconds=self.heartbeat_timeout)
                
                with self.status_lock:
                    online_count = 0
                    offline_count = 0
                    
                    for factory_name, status_info in self.datasite_status.items():
                        last_heartbeat = status_info['last_heartbeat']
                        time_since_heartbeat = current_time - last_heartbeat
                        
                        if time_since_heartbeat > timeout_threshold:
                            if status_info.get('status') != 'offline':
                                self.logger.warning(f"🔴 Datasite {factory_name} marked offline - last heartbeat {time_since_heartbeat.total_seconds():.1f}s ago")
                            status_info['status'] = 'offline'
                            offline_count += 1
                        else:
                            if status_info.get('status') != 'online':
                                self.logger.info(f"🟢 Datasite {factory_name} back online - heartbeat {time_since_heartbeat.total_seconds():.1f}s ago")
                            status_info['status'] = 'online'
                            online_count += 1
                    
                    # Log status summary every 60 seconds
                    if not hasattr(self, '_last_status_log'):
                        self._last_status_log = current_time
                    
                    if (current_time - self._last_status_log).total_seconds() >= 60:
                        self.logger.info(f"📊 Datasite status: {online_count} online, {offline_count} offline")
                        self._last_status_log = current_time
                
                # Every 60 seconds, try to sync with dashboard
                if hasattr(self, '_last_dashboard_sync'):
                    time_since_sync = (current_time - self._last_dashboard_sync).total_seconds()
                    if time_since_sync >= 60:
                        self._sync_with_dashboard()
                        self._last_dashboard_sync = current_time
                else:
                    self._last_dashboard_sync = current_time
                    self._sync_with_dashboard()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}")
                time.sleep(30)
    
    def _sync_with_dashboard(self, dashboard_url: str = "http://localhost:5000"):
        """Sync our datasite information with the dashboard."""
        try:
            import requests
            
            with self.status_lock:
                for factory_name, status_info in self.datasite_status.items():
                    try:
                        # Sync ALL datasites with dashboard (both online and offline)
                        # This ensures dashboard shows TRUE status of ALL connected datasites
                        last_heartbeat = status_info.get('last_heartbeat')
                        registration_data = {
                            "factory_name": factory_name,
                            "factory_hostname": status_info.get('hostname', 'unknown'),
                            "factory_port": status_info.get('port', 8080),
                            "device": status_info.get('device', 'unknown'),
                            "status": "connected" if status_info.get('status') == 'online' else "disconnected",
                            "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
                            "heartbeat_count": status_info.get('heartbeat_count', 0),
                            "client_ip": status_info.get('client_ip', 'unknown')
                        }
                        
                        response = requests.post(
                            f"{dashboard_url}/register_factory",
                            json=registration_data,
                            timeout=3
                        )
                        
                        if response.status_code == 200:
                            self.logger.debug(f"Synced {factory_name} ({status_info.get('status')}) with dashboard")
                        else:
                            self.logger.debug(f"Dashboard sync for {factory_name} returned {response.status_code}")
                    except Exception as sync_error:
                        self.logger.debug(f"Failed to sync {factory_name} with dashboard: {sync_error}")
                        
        except Exception as e:
            self.logger.debug(f"Dashboard sync error: {e}")
    
    def get_available_datasites(self, required_datasites: Optional[List[str]] = None) -> List[str]:
        """
        Get list of currently available datasites.
        
        Args:
            required_datasites: List of specific datasites to check
            
        Returns:
            List of available datasite names
        """
        with self.status_lock:
            available = []
            
            check_list = required_datasites if required_datasites else list(self.datasite_status.keys())
            
            for factory_name in check_list:
                if factory_name in self.datasite_status:
                    if self.datasite_status[factory_name]['status'] == 'online':
                        available.append(factory_name)
            
            return available
    
    def get_datasite_status(self, factory_name: str) -> Optional[Dict]:
        """Get status of specific datasite."""
        with self.status_lock:
            return self.datasite_status.get(factory_name)
    
    def register_datasite(self, factory_name: str, endpoint: Optional[str] = None, status: str = 'online'):
        """
        Manually register a datasite with the heartbeat manager.
        
        Args:
            factory_name: Name of the factory datasite
            endpoint: Optional endpoint URL for the datasite
            status: Initial status ('online', 'offline')
        """
        with self.status_lock:
            self.datasite_status[factory_name] = {
                'last_heartbeat': datetime.now(),
                'status': status,
                'heartbeat_count': 1,
                'endpoint': endpoint or 'Unknown',
                'type': 'Registered'
            }
        self.logger.info(f"Manually registered datasite: {factory_name} with status: {status}")
    
    def get_all_datasite_status(self) -> Dict[str, Dict]:
        """Get status of all datasites with consistent timestamp formatting."""
        with self.status_lock:
            formatted_status = {}
            for factory_name, status in self.datasite_status.items():
                formatted_status[factory_name] = status.copy()
                # Ensure last_heartbeat is always a valid ISO string to prevent JavaScript errors
                if 'last_heartbeat' in formatted_status[factory_name]:
                    if formatted_status[factory_name]['last_heartbeat']:
                        if hasattr(formatted_status[factory_name]['last_heartbeat'], 'isoformat'):
                            formatted_status[factory_name]['last_heartbeat'] = formatted_status[factory_name]['last_heartbeat'].isoformat()
                    else:
                        # Set a default timestamp if None to prevent JavaScript errors
                        formatted_status[factory_name]['last_heartbeat'] = datetime(1970, 1, 1).isoformat()
                else:
                    # Add missing last_heartbeat field with default timestamp
                    formatted_status[factory_name]['last_heartbeat'] = datetime(1970, 1, 1).isoformat()
            return formatted_status
    
    def wait_for_minimum_datasites(self, minimum_count: int = 2, max_wait_seconds: int = 300) -> bool:
        """
        Wait until minimum number of datasites are available.
        
        Args:
            minimum_count: Minimum number of datasites required
            max_wait_seconds: Maximum time to wait
            
        Returns:
            True if minimum datasites available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            available = self.get_available_datasites()
            if len(available) >= minimum_count:
                self.logger.info(f"Minimum datasites available: {available}")
                return True
            
            self.logger.info(f"Waiting for datasites... Currently available: {available}")
            time.sleep(10)
        
        self.logger.warning(f"Timeout waiting for minimum datasites")
        return False
    
    def diagnose_heartbeat_status(self) -> Dict[str, Any]:
        """Comprehensive diagnostic information for debugging heartbeat issues."""
        with self.status_lock:
            current_time = datetime.now()
            timeout_threshold = timedelta(seconds=self.heartbeat_timeout)
            
            diagnosis = {
                'heartbeat_manager': {
                    'running': self.running,
                    'port': self.port,
                    'timeout_seconds': self.heartbeat_timeout,
                    'current_time': current_time.isoformat()
                },
                'datasites': {},
                'summary': {
                    'total_datasites': len(self.datasite_status),
                    'online_datasites': 0,
                    'offline_datasites': 0,
                    'never_contacted': 0
                }
            }
            
            for factory_name, status_info in self.datasite_status.items():
                last_heartbeat = status_info['last_heartbeat']
                time_since_heartbeat = current_time - last_heartbeat
                is_online = time_since_heartbeat <= timeout_threshold
                
                diagnosis['datasites'][factory_name] = {
                    'status': status_info.get('status', 'unknown'),
                    'last_heartbeat': last_heartbeat.isoformat(),
                    'seconds_since_heartbeat': time_since_heartbeat.total_seconds(),
                    'heartbeat_count': status_info.get('heartbeat_count', 0),
                    'hostname': status_info.get('hostname', 'unknown'),
                    'port': status_info.get('port', 'unknown'),
                    'is_within_timeout': is_online,
                    'client_ip': status_info.get('client_ip', 'unknown')
                }
                
                if is_online:
                    diagnosis['summary']['online_datasites'] += 1
                else:
                    diagnosis['summary']['offline_datasites'] += 1
                    
                if status_info.get('heartbeat_count', 0) == 0:
                    diagnosis['summary']['never_contacted'] += 1
            
            return diagnosis
    
    def print_diagnostic_report(self):
        """Print a human-readable diagnostic report."""
        diagnosis = self.diagnose_heartbeat_status()
        
        print("\n" + "="*60)
        print("🔍 HEARTBEAT MANAGER DIAGNOSTIC REPORT")
        print("="*60)
        
        # Manager status
        manager = diagnosis['heartbeat_manager']
        print(f"Manager Status: {'🟢 RUNNING' if manager['running'] else '🔴 STOPPED'}")
        print(f"Port: {manager['port']}")
        print(f"Timeout: {manager['timeout_seconds']}s")
        print(f"Current Time: {manager['current_time']}")
        
        # Summary
        summary = diagnosis['summary']
        print(f"\nDatasite Summary:")
        print(f"  Total: {summary['total_datasites']}")
        print(f"  🟢 Online: {summary['online_datasites']}")
        print(f"  🔴 Offline: {summary['offline_datasites']}")
        print(f"  ❓ Never Contacted: {summary['never_contacted']}")
        
        # Individual datasites
        print(f"\nIndividual Datasites:")
        for name, info in diagnosis['datasites'].items():
            status_icon = "🟢" if info['is_within_timeout'] else "🔴"
            print(f"  {status_icon} {name}:")
            print(f"    Status: {info['status']}")
            print(f"    Last Heartbeat: {info['seconds_since_heartbeat']:.1f}s ago")
            print(f"    Heartbeat Count: {info['heartbeat_count']}")
            print(f"    Source: {info['client_ip']}:{info['port']}")
        
        print("="*60)
    
    def force_sync_all_datasites(self) -> Dict[str, Any]:
        """
        Force immediate sync of ALL datasites with their true status.
        No hardcoded values, only real status based on actual heartbeat data.
        
        Returns:
            Dictionary with sync results
        """
        with self.status_lock:
            current_time = datetime.now()
            timeout_threshold = timedelta(seconds=self.heartbeat_timeout)
            
            sync_results = {
                'updated_count': 0,
                'total_count': len(self.datasite_status),
                'status_changes': []
            }
            
            for factory_name, status_info in self.datasite_status.items():
                old_status = status_info.get('status', 'unknown')
                
                # Calculate TRUE status based on actual heartbeat timing
                last_heartbeat = status_info.get('last_heartbeat')
                if last_heartbeat:
                    time_since_heartbeat = current_time - last_heartbeat
                    true_status = 'online' if time_since_heartbeat <= timeout_threshold else 'offline'
                else:
                    true_status = 'offline'  # Never received heartbeat
                
                # Update status if it changed
                if old_status != true_status:
                    status_info['status'] = true_status
                    status_info['last_update'] = current_time.isoformat()
                    
                    sync_results['status_changes'].append({
                        'datasite': factory_name,
                        'old_status': old_status,
                        'new_status': true_status,
                        'last_heartbeat_seconds_ago': (current_time - last_heartbeat).total_seconds() if last_heartbeat else None
                    })
                    
                    self.logger.info(f"Status updated: {factory_name} {old_status} → {true_status}")
                
                sync_results['updated_count'] += 1
            
            return sync_results
    
    def get_real_time_status_summary(self) -> Dict[str, Any]:
        """
        Get real-time status summary with TRUE datasite availability.
        No fallback values - only actual status based on heartbeat data.
        
        Returns:
            Dictionary with comprehensive status summary
        """
        with self.status_lock:
            current_time = datetime.now()
            timeout_threshold = timedelta(seconds=self.heartbeat_timeout)
            
            summary = {
                'timestamp': current_time.isoformat(),
                'total_registered': len(self.datasite_status),
                'online_count': 0,
                'offline_count': 0,
                'never_contacted_count': 0,
                'datasites': {}
            }
            
            for factory_name, status_info in self.datasite_status.items():
                last_heartbeat = status_info.get('last_heartbeat')
                heartbeat_count = status_info.get('heartbeat_count', 0)
                
                # Calculate real-time status
                if last_heartbeat:
                    time_since_heartbeat = current_time - last_heartbeat
                    is_online = time_since_heartbeat <= timeout_threshold
                    seconds_ago = time_since_heartbeat.total_seconds()
                else:
                    is_online = False
                    seconds_ago = None
                
                true_status = 'online' if is_online else 'offline'
                
                # Update counts
                if is_online:
                    summary['online_count'] += 1
                else:
                    summary['offline_count'] += 1
                    
                if heartbeat_count == 0:
                    summary['never_contacted_count'] += 1
                
                # Store detailed datasite info
                summary['datasites'][factory_name] = {
                    'status': true_status,
                    'last_heartbeat': last_heartbeat.isoformat() if last_heartbeat else None,
                    'seconds_since_heartbeat': seconds_ago,
                    'heartbeat_count': heartbeat_count,
                    'hostname': status_info.get('hostname', 'unknown'),
                    'port': status_info.get('port', 'unknown'),
                    'client_ip': status_info.get('client_ip', 'unknown'),
                    'within_timeout': is_online
                }
            
            return summary
    
    def print_real_time_status(self):
        """Print real-time status of all datasites with TRUE availability info."""
        summary = self.get_real_time_status_summary()
        
        print("\n" + "="*70)
        print("🔍 REAL-TIME DATASITE STATUS (TRUE AVAILABILITY)")
        print("="*70)
        print(f"Total Registered: {summary['total_registered']}")
        print(f"🟢 Online: {summary['online_count']}")
        print(f"🔴 Offline: {summary['offline_count']}")
        print(f"❓ Never Contacted: {summary['never_contacted_count']}")
        print(f"Timestamp: {summary['timestamp']}")
        
        if summary['datasites']:
            print(f"\nDetailed Status:")
            for name, info in summary['datasites'].items():
                status_icon = "🟢" if info['status'] == 'online' else "🔴"
                last_contact = f"{info['seconds_since_heartbeat']:.1f}s ago" if info['seconds_since_heartbeat'] else "Never"
                
                print(f"  {status_icon} {name}")
                print(f"     Status: {info['status']}")
                print(f"     Last Contact: {last_contact}")
                print(f"     Heartbeats: {info['heartbeat_count']}")
                print(f"     Source: {info['client_ip']}:{info['port']}")
        
        print("="*70)

    def are_datasites_healthy(self, datasite_names: List[str], 
                             required_ratio: float = 0.5) -> Tuple[bool, List[str]]:
        """
        Check if enough datasites are healthy for continuing experiment.
        
        Args:
            datasite_names: List of datasites to check
            required_ratio: Minimum ratio of healthy datasites required
            
        Returns:
            Tuple of (is_healthy, list_of_available_datasites)
        """
        available = self.get_available_datasites(datasite_names)
        healthy_ratio = len(available) / len(datasite_names) if datasite_names else 0
        
        is_healthy = healthy_ratio >= required_ratio and len(available) >= 2
        
        return is_healthy, available
# Global heartbeat manager instance
_heartbeat_manager = None

def get_heartbeat_manager() -> HeartbeatManager:
    """Get global heartbeat manager instance."""
    global _heartbeat_manager
    if _heartbeat_manager is None:
        _heartbeat_manager = HeartbeatManager()
    return _heartbeat_manager

def start_heartbeat_manager():
    """Start global heartbeat manager."""
    manager = get_heartbeat_manager()
    manager.start()
    return manager

def stop_heartbeat_manager():
    """Stop global heartbeat manager."""
    global _heartbeat_manager
    if _heartbeat_manager:
        _heartbeat_manager.stop()
        _heartbeat_manager = None
