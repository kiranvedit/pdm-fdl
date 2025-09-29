#!/usr/bin/env python3
"""
Quick Datasite Connectivity Test
===============================

Simple script to test connectivity to your configured PySyft datasites.
Uses the existing YAML config: Step1-Experiment/NetworkFed/config/datasite_configs.yaml

Usage:
    python test_datasite_connectivity.py
    python test_datasite_connectivity.py --verbose
    python test_datasite_connectivity.py --datasite factory_01
"""

import argparse
import socket
import sys
import time
from pathlib import Path
import requests

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("Installing PyYAML...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML"])
    import yaml
    YAML_AVAILABLE = True

try:
    import syft as sy
    SYFT_AVAILABLE = True
except ImportError:
    SYFT_AVAILABLE = False

def load_datasite_config(config_path: str = None):
    """Load datasite configuration from YAML file."""
    if config_path is None:
        config_path = "Step1-Experiment/NetworkFed/config/datasite_configs.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        datasites = []
        for site_id, site_config in config['datasites'].items():
            datasite = {
                'site_id': site_id,
                'name': site_config.get('site_name', site_id),
                'hostname': site_config.get('hostname', 'localhost'),
                'port': site_config.get('port', 8080),
                'email': site_config.get('admin_email', 'admin@pdm-factory.com'),
                'password': site_config.get('admin_password', 'password'),
                'url': f"http://{site_config.get('hostname', 'localhost')}:{site_config.get('port', 8080)}"
            }
            datasites.append(datasite)
        
        return datasites
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return []

def test_basic_connectivity(hostname: str, port: int, timeout: int = 5):
    """Test basic network connectivity."""
    try:
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            response_time = (time.time() - start_time) * 1000
            return True, f"{response_time:.2f}ms"
        else:
            return False, f"Connection failed (error {result})"
    except Exception as e:
        return False, str(e)

def test_http_response(url: str, timeout: int = 5):
    """Test HTTP response."""
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        return True, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_pysyft_connection(url: str, email: str, password: str):
    """Test PySyft connection."""
    if not SYFT_AVAILABLE:
        return False, "PySyft not available"
    
    try:
        client = sy.login(url=url, email=email, password=password)
        return True, "Authentication successful"
    except Exception as e:
        return False, f"PySyft error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Test datasite connectivity")
    parser.add_argument('--config', default=None, help="Config file path")
    parser.add_argument('--datasite', help="Test specific datasite only")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    print("🔗 DATASITE CONNECTIVITY TEST")
    print("=" * 50)
    
    # Load configuration
    datasites = load_datasite_config(args.config)
    if not datasites:
        print("❌ No datasites loaded from configuration")
        return 1
    
    # Filter to specific datasite if requested
    if args.datasite:
        datasites = [d for d in datasites if d['site_id'] == args.datasite]
        if not datasites:
            print(f"❌ Datasite '{args.datasite}' not found in configuration")
            return 1
    
    print(f"Testing {len(datasites)} datasite(s)...")
    print()
    
    success_count = 0
    total_count = len(datasites)
    
    for datasite in datasites:
        site_id = datasite['site_id']
        hostname = datasite['hostname']
        port = datasite['port']
        url = datasite['url']
        email = datasite['email']
        password = datasite['password']
        
        print(f"🏭 {site_id} ({hostname}:{port})")
        
        # Test 1: Network connectivity
        network_ok, network_msg = test_basic_connectivity(hostname, port)
        status = "✅" if network_ok else "❌"
        print(f"  Network: {status} {network_msg}")
        
        if args.verbose or not network_ok:
            # Test 2: HTTP response
            http_ok, http_msg = test_http_response(url)
            status = "✅" if http_ok else "❌"
            print(f"  HTTP:    {status} {http_msg}")
            
            # Test 3: PySyft connection
            if network_ok:
                syft_ok, syft_msg = test_pysyft_connection(url, email, password)
                status = "✅" if syft_ok else "❌"
                print(f"  PySyft:  {status} {syft_msg}")
                
                if network_ok and syft_ok:
                    success_count += 1
            else:
                print(f"  PySyft:  ⏭️  Skipped (network failed)")
        else:
            success_count += 1
        
        print()
    
    # Summary
    print("📊 SUMMARY")
    print("-" * 20)
    print(f"Total datasites: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 All datasites are accessible!")
        return 0
    elif success_count > 0:
        print("⚠️  Partial connectivity")
        return 2
    else:
        print("❌ No datasites accessible")
        return 1

if __name__ == "__main__":
    sys.exit(main())
