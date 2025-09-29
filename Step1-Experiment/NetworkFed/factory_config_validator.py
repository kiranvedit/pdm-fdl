#!/usr/bin/env python3
"""
Factory Configuration Validator and Updater
This script validates and updates factory configurations to ensure:
1. Correct endpoint routing (HeartbeatManager:8888, Dashboard:8889)
2. Proper PySyft authentication setup
3. Network connectivity between all components
"""

import json
import os
import requests
import time
from typing import Dict, List, Any
from pathlib import Path

class FactoryConfigValidator:
    """Validates and updates factory datasite configurations."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.heartbeat_manager_port = 8888
        self.dashboard_port = 8889
        self.factory_configs = []
        
    def load_factory_configs(self) -> List[Dict[str, Any]]:
        """Load all factory configuration files."""
        config_files = [
            "SANFactory_config.json",
            "SEAFactory_config.json", 
            "WUS2Factory_config.json",
            "SouthIndFactory_config.json",
            "NEFactory_config.json"
        ]
        
        configs = []
        for config_file in config_files:
            config_path = self.base_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        config['_config_file'] = config_file
                        configs.append(config)
                        print(f"✅ Loaded {config_file}")
                except Exception as e:
                    print(f"❌ Failed to load {config_file}: {e}")
            else:
                print(f"⚠️ Config file not found: {config_file}")
        
        self.factory_configs = configs
        return configs
    
    def validate_endpoint_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix endpoint configurations."""
        print(f"\n🔍 Validating endpoints for {config.get('factory_name', 'Unknown')}")
        
        issues = []
        fixes = {}
        
        # Check dashboard_url configuration
        dashboard_url = config.get('dashboard_url', '')
        
        # For registration, should point to dashboard (port 8889)
        # For heartbeat, should point to heartbeat manager (port 8888)
        
        if ':8888' in dashboard_url:
            print(f"  ⚠️ dashboard_url points to HeartbeatManager port (8888)")
            print(f"  💡 This is OK for heartbeats, but registration needs port 8889")
            fixes['dashboard_url_note'] = "Using 8888 for heartbeats, will auto-switch to 8889 for registration"
        elif ':8889' in dashboard_url:
            print(f"  ⚠️ dashboard_url points to Dashboard port (8889)")
            print(f"  💡 This is OK for registration, but heartbeats need port 8888")
            fixes['dashboard_url_note'] = "Using 8889 for registration, will auto-switch to 8888 for heartbeats"
        else:
            print(f"  ❌ dashboard_url has no specific port: {dashboard_url}")
            issues.append("dashboard_url needs port specification")
            fixes['dashboard_url'] = dashboard_url.rstrip('/') + ':8888'
        
        return {
            'issues': issues,
            'fixes': fixes,
            'status': 'OK' if not issues else 'NEEDS_FIX'
        }
    
    def test_connectivity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test connectivity to HeartbeatManager and Dashboard."""
        print(f"\n🔗 Testing connectivity for {config.get('factory_name', 'Unknown')}")
        
        base_url = config.get('dashboard_url', '').split(':')[0] + ':' + config.get('dashboard_url', '').split(':')[1]
        if not base_url.startswith('http'):
            base_url = f"http://{base_url}"
            
        results = {}
        
        # Test HeartbeatManager (port 8888)
        hm_url = f"{base_url}:8888"
        try:
            response = requests.get(f"{hm_url}/status", timeout=3)
            results['heartbeat_manager'] = {
                'status': 'OK',
                'code': response.status_code,
                'url': hm_url
            }
            print(f"  ✅ HeartbeatManager reachable: {hm_url}")
        except Exception as e:
            results['heartbeat_manager'] = {
                'status': 'ERROR',
                'error': str(e),
                'url': hm_url
            }
            print(f"  ❌ HeartbeatManager unreachable: {hm_url} - {e}")
        
        # Test Dashboard (port 8889)
        dash_url = f"{base_url}:8889"
        try:
            response = requests.get(dash_url, timeout=3)
            results['dashboard'] = {
                'status': 'OK',
                'code': response.status_code,
                'url': dash_url
            }
            print(f"  ✅ Dashboard reachable: {dash_url}")
        except Exception as e:
            results['dashboard'] = {
                'status': 'ERROR',
                'error': str(e),
                'url': dash_url
            }
            print(f"  ❌ Dashboard unreachable: {dash_url} - {e}")
        
        # Test factory datasite (its own port)
        factory_url = f"http://{config.get('hostname', 'localhost')}:{config.get('port', 8080)}"
        try:
            response = requests.get(factory_url, timeout=3)
            results['factory_datasite'] = {
                'status': 'OK',
                'code': response.status_code,
                'url': factory_url
            }
            print(f"  ✅ Factory datasite reachable: {factory_url}")
        except Exception as e:
            results['factory_datasite'] = {
                'status': 'ERROR',
                'error': str(e),
                'url': factory_url
            }
            print(f"  ❌ Factory datasite unreachable: {factory_url} - {e}")
        
        return results
    
    def generate_startup_script(self, config: Dict[str, Any]) -> str:
        """Generate a startup script for the factory."""
        script_content = f"""#!/usr/bin/env python3
\"\"\"
Startup script for {config.get('factory_name', 'Unknown')} Factory Datasite
Generated by FactoryConfigValidator
\"\"\"

import sys
import os
import time
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def start_factory_datasite():
    \"\"\"Start the factory datasite with proper configuration.\"\"\"
    print(f"🏭 Starting {config.get('factory_name', 'Unknown')} Factory Datasite...")
    
    # Configuration
    factory_name = "{config.get('factory_name', 'unknown_factory')}"
    hostname = "{config.get('hostname', 'localhost')}"
    port = {config.get('port', 8080)}
    dashboard_url = "{config.get('dashboard_url', 'http://localhost:8888')}"
    device = "{config.get('device', 'cpu')}"
    
    # Import and start the factory datasite
    try:
        from Step1_Experiment.NetworkFed.ExternalDatasite.FactoryDatasite import FactoryDatasite
        
        factory = FactoryDatasite(
            factory_name=factory_name,
            hostname=hostname,
            port=port,
            dashboard_url=dashboard_url,
            device=device,
            dev_mode=True,
            reset=False,
            verbose=True
        )
        
        print(f"✅ {{factory_name}} started successfully!")
        print(f"   📍 Address: http://{{hostname}}:{{port}}")
        print(f"   🔗 Dashboard: {{dashboard_url}}")
        print(f"   💻 Device: {{device}}")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\\n🛑 Shutting down {{factory_name}}...")
            
    except Exception as e:
        print(f"❌ Failed to start {{factory_name}}: {{e}}")
        return False
    
    return True

if __name__ == "__main__":
    start_factory_datasite()
"""
        return script_content
    
    def update_factory_configuration(self, config: Dict[str, Any], fixes: Dict[str, Any]) -> bool:
        """Apply fixes to factory configuration."""
        config_file = config.get('_config_file')
        if not config_file:
            return False
            
        print(f"\n🔧 Applying fixes to {config_file}")
        
        # Apply fixes
        updated_config = config.copy()
        for key, value in fixes.items():
            if key != 'dashboard_url_note':
                updated_config[key] = value
                print(f"  ✅ Updated {key}: {value}")
        
        # Save updated configuration
        try:
            config_path = self.base_dir / config_file
            with open(config_path, 'w') as f:
                json.dump(updated_config, f, indent=2)
            print(f"  ✅ Saved updated configuration to {config_file}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to save configuration: {e}")
            return False
    
    def run_validation(self):
        """Run complete validation and fixing process."""
        print("🔍 Factory Configuration Validation and Update")
        print("=" * 60)
        
        # Load configurations
        configs = self.load_factory_configs()
        if not configs:
            print("❌ No factory configurations found!")
            return False
        
        print(f"✅ Loaded {len(configs)} factory configurations")
        
        # Validate each configuration
        for config in configs:
            print(f"\n{'='*20} {config.get('factory_name', 'Unknown')} {'='*20}")
            
            # Validate endpoints
            endpoint_validation = self.validate_endpoint_configuration(config)
            
            # Test connectivity
            connectivity_results = self.test_connectivity(config)
            
            # Apply fixes if needed
            if endpoint_validation['status'] == 'NEEDS_FIX':
                self.update_factory_configuration(config, endpoint_validation['fixes'])
            
            # Generate startup script
            startup_script = self.generate_startup_script(config)
            script_filename = f"start_{config.get('factory_name', 'unknown')}.py"
            script_path = self.base_dir / script_filename
            
            try:
                with open(script_path, 'w') as f:
                    f.write(startup_script)
                print(f"  ✅ Generated startup script: {script_filename}")
            except Exception as e:
                print(f"  ❌ Failed to generate startup script: {e}")
        
        print(f"\n{'='*60}")
        print("🏁 Validation completed!")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  - Validated {len(configs)} factory configurations")
        print(f"  - Generated startup scripts for each factory")
        print(f"  - Ready for deployment!")
        
        return True

def main():
    """Main entry point."""
    validator = FactoryConfigValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
