#!/usr/bin/env python3
"""
Factory Datasite Configuration Setup
===================================
Easy setup script to configure different factory datasites from templates.

Usage:
  python setup_factory_config.py --factory SANFactory    # Setup SANFactory configuration
  python setup_factory_config.py --factory SEAFactory    # Setup SEAFactory configuration  
  python setup_factory_config.py --list                  # List available factory templates
  python setup_factory_config.py --custom                # Create custom configuration
"""

import json
import shutil
import argparse
from pathlib import Path

class FactoryConfigSetup:
    def __init__(self):
        self.config_dir = Path(__file__).parent / "config_templates"
        self.target_config = Path(__file__).parent / "service_config.json"
        
    def list_available_factories(self):
        """List available factory configuration templates"""
        print("🏭 Available Factory Configuration Templates:")
        print("=" * 50)
        
        if not self.config_dir.exists():
            print("❌ No configuration templates found")
            return []
        
        factories = []
        for config_file in self.config_dir.glob("*_config.json"):
            factory_name = config_file.stem.replace("_config", "")
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    hostname = config.get('hostname', 'unknown')
                    port = config.get('port', 'unknown')
                    print(f"   🔧 {factory_name:<15} (IP: {hostname}, Port: {port})")
                    factories.append(factory_name)
            except Exception as e:
                print(f"   ❌ {factory_name:<15} (Error: {e})")
        
        return factories
    
    def load_factory_config(self, factory_name):
        """Load configuration for a specific factory"""
        config_file = self.config_dir / f"{factory_name}_config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration template for {factory_name} not found")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def apply_factory_config(self, factory_name):
        """Apply factory configuration as active service configuration"""
        try:
            config = self.load_factory_config(factory_name)
            
            # Write to service_config.json
            with open(self.target_config, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"✅ Applied {factory_name} configuration successfully!")
            print(f"📝 Configuration written to: {self.target_config}")
            print(f"🏭 Factory Name: {config.get('factory_name')}")
            print(f"🌐 Hostname: {config.get('hostname')}")
            print(f"🔌 Port: {config.get('port')}")
            print(f"📡 Dashboard URL: {config.get('dashboard_url')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to apply {factory_name} configuration: {e}")
            return False
    
    def create_custom_config(self):
        """Interactive creation of custom factory configuration"""
        print("🔧 Creating Custom Factory Configuration")
        print("=" * 50)
        
        config = {}
        
        # Basic configuration
        config['factory_name'] = input("Factory Name (e.g., MyFactory): ").strip()
        config['hostname'] = input("Hostname/IP (e.g., 192.168.1.100): ").strip()
        config['port'] = int(input("Port (e.g., 8080): ").strip())
        config['dashboard_url'] = input("Dashboard URL (e.g., http://192.168.1.1:8888): ").strip()
        
        # Optional settings
        config['dev_mode'] = input("Development mode? (y/n): ").strip().lower() == 'y'
        config['reset'] = input("Reset datasite on startup? (y/n): ").strip().lower() == 'y'
        config['verbose'] = input("Verbose logging? (y/n): ").strip().lower() == 'y'
        
        # Default settings
        config['python_executable'] = "python"
        config['script_path'] = "FactoryDatasite.py"
        config['restart_on_failure'] = True
        config['restart_delay_seconds'] = 5
        config['max_restart_attempts'] = 10
        config['heartbeat_interval'] = 30
        
        # Save configuration
        try:
            with open(self.target_config, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"\n✅ Custom configuration created successfully!")
            print(f"📝 Configuration saved to: {self.target_config}")
            
            # Optionally save as template
            save_template = input(f"\nSave as template for future use? (y/n): ").strip().lower() == 'y'
            if save_template:
                template_name = config['factory_name']
                template_file = self.config_dir / f"{template_name}_config.json"
                self.config_dir.mkdir(exist_ok=True)
                shutil.copy2(self.target_config, template_file)
                print(f"📚 Template saved as: {template_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create custom configuration: {e}")
            return False
    
    def show_current_config(self):
        """Show current active configuration"""
        if not self.target_config.exists():
            print("❌ No active configuration found")
            return
        
        try:
            with open(self.target_config, 'r') as f:
                config = json.load(f)
            
            print("📋 Current Active Configuration:")
            print("=" * 30)
            print(f"🏭 Factory Name: {config.get('factory_name')}")
            print(f"🌐 Hostname: {config.get('hostname')}")
            print(f"🔌 Port: {config.get('port')}")
            print(f"📡 Dashboard URL: {config.get('dashboard_url')}")
            print(f"🐛 Dev Mode: {config.get('dev_mode')}")
            print(f"🔄 Reset: {config.get('reset')}")
            print(f"📢 Verbose: {config.get('verbose')}")
            
        except Exception as e:
            print(f"❌ Error reading current configuration: {e}")

def main():
    parser = argparse.ArgumentParser(description="Factory Datasite Configuration Setup")
    parser.add_argument("--factory", help="Apply configuration for specific factory")
    parser.add_argument("--list", action="store_true", help="List available factory templates")
    parser.add_argument("--custom", action="store_true", help="Create custom configuration")
    parser.add_argument("--current", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    setup = FactoryConfigSetup()
    
    if args.list:
        setup.list_available_factories()
    elif args.factory:
        setup.apply_factory_config(args.factory)
    elif args.custom:
        setup.create_custom_config()
    elif args.current:
        setup.show_current_config()
    else:
        print("Factory Datasite Configuration Setup")
        print("=" * 40)
        setup.show_current_config()
        print("\nAvailable commands:")
        print("  --list     List available factory templates")
        print("  --factory  Apply specific factory configuration")
        print("  --custom   Create custom configuration")
        print("  --current  Show current configuration")
        print("\nExample:")
        print("  python setup_factory_config.py --factory SANFactory")

if __name__ == "__main__":
    main()
