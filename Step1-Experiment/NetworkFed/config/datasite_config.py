"""
Datasite Configuration Management
Handles configuration for external PySyft datasites
"""

import json
import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class DatasiteConfig:
    """Simple datasite configuration class"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.id = config_dict.get('id', config_dict.get('site_id', ''))
        self.name = config_dict.get('name', config_dict.get('site_name', ''))
        self.site_name = config_dict.get('site_name', config_dict.get('name', ''))
        self.hostname = config_dict.get('hostname', config_dict.get('host', 'localhost'))
        self.host = self.hostname  # Backward compatibility
        self.port = config_dict.get('port', 8080)
        self.admin_email = config_dict.get('admin_email', 'admin@factory.com')
        self.admin_password = config_dict.get('admin_password', 'password')
        self.data_path = config_dict.get('data_path', '')
        self.secure = config_dict.get('secure', False)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config


class DatasiteConfigManager:
    """Manages configuration for external datasites"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = "config/datasite_configs.yaml"
        self.config_path = config_path
        self.config = None
        self.datasites = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load datasite configuration from YAML or JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Datasite configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)
                
        # Load datasites (handle both dict and list formats)
        self.datasites = []
        datasites_config = self.config.get('datasites', {})
        
        if isinstance(datasites_config, dict):
            # YAML format: datasites is a dict with site names as keys
            for site_name, datasite_config in datasites_config.items():
                # Ensure the config has the site_id if not present
                if 'site_id' not in datasite_config:
                    datasite_config['site_id'] = site_name
                if 'id' not in datasite_config:
                    datasite_config['id'] = site_name
                if 'name' not in datasite_config:
                    datasite_config['name'] = datasite_config.get('site_name', site_name)
                self.datasites.append(DatasiteConfig(datasite_config))
        elif isinstance(datasites_config, list):
            # JSON format: datasites is a list
            for datasite_config in datasites_config:
                self.datasites.append(DatasiteConfig(datasite_config))
            
        return self.config
    
    def get_datasite_configs(self) -> List[Dict[str, Any]]:
        """Get list of datasite configurations"""
        if self.config is None:
            self.load_config()
            
        datasites_config = self.config.get('datasites', {}) if self.config else {}
        
        if isinstance(datasites_config, dict):
            # Convert dict format to list format
            result = []
            for site_name, datasite_config in datasites_config.items():
                config_copy = datasite_config.copy()
                if 'site_id' not in config_copy:
                    config_copy['site_id'] = site_name
                if 'id' not in config_copy:
                    config_copy['id'] = site_name
                if 'name' not in config_copy:
                    config_copy['name'] = config_copy.get('site_name', site_name)
                result.append(config_copy)
            return result
        elif isinstance(datasites_config, list):
            return datasites_config
        else:
            return []
    
    def validate_config(self) -> bool:
        """Validate datasite configuration"""
        try:
            config = self.load_config()
            
            if not config:
                return False
                
            # Check required fields
            if 'datasites' not in config:
                return False
                
            datasites_config = config['datasites']
            if isinstance(datasites_config, dict):
                # YAML format: datasites is a dict
                for site_name, datasite in datasites_config.items():
                    required_fields = ['hostname', 'port']  # site_id is auto-generated
                    for field in required_fields:
                        if field not in datasite:
                            return False
            elif isinstance(datasites_config, list):
                # JSON format: datasites is a list
                for datasite in datasites_config:
                    required_fields = ['id', 'name', 'host', 'port']
                    for field in required_fields:
                        if field not in datasite:
                            return False
            else:
                return False
                        
            return True
        except Exception:
            return False

    def get_datasite(self, site_id: str) -> Optional[DatasiteConfig]:
        """Get datasite configuration by site_id"""
        if self.config is None:
            self.load_config()
            
        for datasite in self.datasites:
            if datasite.id == site_id:
                return datasite
        return None

    def get_datasite_dict(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Get datasite configuration as dictionary by site_id"""
        datasite = self.get_datasite(site_id)
        return datasite.to_dict() if datasite else None


def load_datasite_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load datasite configuration from file"""
    if config_path is None:
        config_path = "config/datasite_configs.yaml"
    manager = DatasiteConfigManager(config_path)
    return manager.load_config()


def get_default_datasite_config() -> Dict[str, Any]:
    """Get default datasite configuration for local development"""
    return {
        "datasites": [
            {
                "id": "datasite_1",
                "name": "Local Datasite 1",
                "host": "localhost",
                "port": 8081,
                "data_path": "data/site1",
                "secure": False
            },
            {
                "id": "datasite_2", 
                "name": "Local Datasite 2",
                "host": "localhost",
                "port": 8082,
                "data_path": "data/site2",
                "secure": False
            },
            {
                "id": "datasite_3",
                "name": "Local Datasite 3", 
                "host": "localhost",
                "port": 8083,
                "data_path": "data/site3",
                "secure": False
            }
        ]
    }


def create_default_config_file(config_path: Optional[str] = None):
    """Create default configuration file"""
    if config_path is None:
        config_path = "config/datasite_configs.yaml"
        
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config = get_default_datasite_config()
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, indent=2, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)
        
    return config_path
