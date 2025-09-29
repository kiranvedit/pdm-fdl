# Federated Learning Configuration Management

This directory contains configuration files and utilities for managing external datasite connections in the federated learning framework. It provides centralized configuration management for PySyft datasites across distributed industrial environments.

## Directory Contents

### Configuration Files
- **`datasite_configs.yaml`** - Main configuration file for external PySyft datasites
- **`datasite_config.py`** - Configuration management utilities and classes

### Connectivity Testing
- **`../test_datasite_connectivity.py`** - Datasite connectivity testing script

---

## 1. Datasite Configuration (`datasite_configs.yaml`)

### What is Datasite Configuration?
The YAML configuration file defines connection parameters for external PySyft datasites running on different edge devices or virtual machines or physical locations. It provides a centralized way to manage multiple industrial sites in federated learning experiments.

### Configuration Structure
```yaml
datasites:
  factory_01:
    site_id: factory_01
    hostname: localhost           # Change to actual VM IP/hostname
    port: 8081
    admin_email: admin@pdm-factory.com
    admin_password: password
    site_name: Factory_01
  
  factory_02:
    site_id: factory_02
    hostname: localhost           # Change to actual VM IP/hostname
    port: 8082
    admin_email: admin@pdm-factory.com
    admin_password: password
    site_name: Factory_02
  
  factory_03:
    site_id: factory_03
    hostname: localhost           # Change to actual VM IP/hostname
    port: 8083
    admin_email: admin@pdm-factory.com
    admin_password: password
    site_name: Factory_03
```

### Configuration Parameters

#### **Site Identification**
- **`site_id`**: Unique identifier for the datasite (used in federated learning algorithms)
- **`site_name`**: Human-readable name for the industrial site
- **`hostname`**: IP address or hostname of the datasite server
- **`port`**: Port number where the PySyft datasite is running

#### **Authentication**
- **`admin_email`**: Administrator email for PySyft authentication
- **`admin_password`**: Administrator password for secure access

### Production Configuration Example
```yaml
datasites:
  manufacturing_plant_detroit:
    site_id: manufacturing_plant_detroit
    hostname: 192.168.1.100        # Actual VM IP
    port: 8081
    admin_email: admin@manufacturing.com
    admin_password: secure_password_123
    site_name: Detroit_Manufacturing_Plant
  
  assembly_line_chicago:
    site_id: assembly_line_chicago
    hostname: 192.168.1.101        # Actual VM IP
    port: 8082
    admin_email: admin@manufacturing.com
    admin_password: secure_password_123
    site_name: Chicago_Assembly_Line
```

### Setup Instructions

#### **1. Update Hostnames**
Replace `localhost` with actual VM IP addresses:
```yaml
hostname: 192.168.1.100  # Change from localhost
```

#### **2. Configure Ports**
Ensure ports match your datasite deployments:
```yaml
port: 8081  # Must match datasite startup port
```

#### **3. Set Credentials**
Update authentication credentials:
```yaml
admin_email: your-admin@company.com
admin_password: your-secure-password
```

#### **4. Add Additional Sites**
Extend configuration for more industrial locations:
```yaml
  factory_04:
    site_id: factory_04
    hostname: 192.168.1.103
    port: 8084
    admin_email: admin@pdm-factory.com
    admin_password: password
    site_name: Factory_04
```

---

## 2. Configuration Management (`datasite_config.py`)

### What is Configuration Management?
Python utilities for loading, validating, and managing datasite configurations programmatically. Provides a consistent interface for accessing configuration data across the federated learning framework.

### Core Classes

#### **DatasiteConfig Class**
```python
class DatasiteConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.id = config_dict.get('id', config_dict.get('site_id', ''))
        self.hostname = config_dict.get('hostname', 'localhost')
        self.port = config_dict.get('port', 8080)
        self.admin_email = config_dict.get('admin_email', 'admin@factory.com')
        self.admin_password = config_dict.get('admin_password', 'password')
```

**Features:**
- Flexible configuration parsing
- Backward compatibility with different naming conventions
- Default value handling for missing parameters

#### **DatasiteConfigManager Class**
```python
class DatasiteConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = "config/datasite_configs.yaml"
```

**Key Methods:**
- `load_config()`: Load YAML or JSON configuration files
- `validate_config()`: Validate configuration completeness
- `get_datasite_configs()`: Get list of all datasite configurations
- `get_datasite(site_id)`: Get specific datasite configuration

### Usage Examples

#### **Basic Configuration Loading**
```python
from config.datasite_config import DatasiteConfigManager

# Load default configuration
manager = DatasiteConfigManager()
config = manager.load_config()

# Get all datasite configurations
datasites = manager.get_datasite_configs()

# Get specific datasite
factory_01 = manager.get_datasite('factory_01')
print(f"Connecting to {factory_01.hostname}:{factory_01.port}")
```

#### **Custom Configuration Path**
```python
# Load from custom path
manager = DatasiteConfigManager("custom/path/datasites.yaml")
config = manager.load_config()
```

#### **Configuration Validation**
```python
# Validate configuration before use
if manager.validate_config():
    print("✅ Configuration is valid")
    datasites = manager.get_datasite_configs()
else:
    print("❌ Configuration validation failed")
```

### Multi-Format Support

#### **YAML Format (Recommended)**
```yaml
datasites:
  factory_01:
    hostname: localhost
    port: 8081
    admin_email: admin@factory.com
```

#### **JSON Format**
```json
{
  "datasites": [
    {
      "id": "factory_01",
      "host": "localhost",
      "port": 8081,
      "admin_email": "admin@factory.com"
    }
  ]
}
```

#### **Default Configuration Generation**
```python
from config.datasite_config import create_default_config_file

# Create default configuration file
config_path = create_default_config_file("config/datasite_configs.yaml")
print(f"Created default config at: {config_path}")
```

---

## 3. Connectivity Testing (`test_datasite_connectivity.py`)

### What is Connectivity Testing?
A comprehensive testing script that verifies network connectivity, HTTP response, and PySyft authentication with configured datasites. Essential for validating distributed federated learning setup.

### Testing Capabilities

#### **Multi-Level Testing**
1. **Network Connectivity**: TCP socket connection test
2. **HTTP Response**: Web service availability test  
3. **PySyft Authentication**: Full authentication workflow test

#### **Test Execution Modes**

**Test All Datasites (Default):**
```bash
python test_datasite_connectivity.py
```

**Verbose Testing:**
```bash
python test_datasite_connectivity.py --verbose
```

**Test Specific Datasite:**
```bash
python test_datasite_connectivity.py --datasite factory_01
```

**Custom Configuration:**
```bash
python test_datasite_connectivity.py --config custom/path/config.yaml
```

### Test Output Examples

#### **Basic Test Output**
```
🔗 DATASITE CONNECTIVITY TEST
==================================================
Testing 3 datasite(s)...

🏭 factory_01 (localhost:8081)
  Network: ✅ 2.45ms

🏭 factory_02 (localhost:8082)
  Network: ✅ 1.23ms

🏭 factory_03 (localhost:8083)
  Network: ❌ Connection failed (error 111)

📊 SUMMARY
--------------------
Total datasites: 3
Successful: 2
Failed: 1
⚠️ Partial connectivity
```

#### **Verbose Test Output**
```
🏭 factory_01 (localhost:8081)
  Network: ✅ 2.45ms
  HTTP:    ✅ HTTP 200
  PySyft:  ✅ Authentication successful

🏭 factory_02 (localhost:8082)
  Network: ✅ 1.23ms
  HTTP:    ✅ HTTP 200
  PySyft:  ❌ PySyft error: Authentication failed

🏭 factory_03 (localhost:8083)
  Network: ❌ Connection failed (error 111)
  HTTP:    ❌ Connection refused
  PySyft:  ⏭️ Skipped (network failed)
```

### Test Functions

#### **Network Connectivity Test**
```python
def test_basic_connectivity(hostname: str, port: int, timeout: int = 5):
    """Test basic network connectivity using TCP socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()
        return result == 0, response_time
    except Exception as e:
        return False, str(e)
```

#### **HTTP Response Test**
```python
def test_http_response(url: str, timeout: int = 5):
    """Test HTTP response from datasite web service."""
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        return True, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)
```

#### **PySyft Authentication Test**
```python
def test_pysyft_connection(url: str, email: str, password: str):
    """Test full PySyft authentication workflow."""
    try:
        client = sy.login(url=url, email=email, password=password)
        return True, "Authentication successful"
    except Exception as e:
        return False, f"PySyft error: {e}"
```

### Return Codes
- **0**: All datasites accessible
- **1**: No datasites accessible  
- **2**: Partial connectivity (some datasites accessible)

---

## 4. Integration with Federated Learning Framework

### Experiment Integration

#### **Loading Configuration in Experiments**
```python
from config.datasite_config import DatasiteConfigManager

# Initialize configuration manager
config_manager = DatasiteConfigManager()
datasites = config_manager.get_datasite_configs()

# Use in federated experiments
for datasite in datasites:
    print(f"Connecting to {datasite['site_id']} at {datasite['hostname']}:{datasite['port']}")
    # Initialize PySyft client connection
    client = sy.login(
        url=f"http://{datasite['hostname']}:{datasite['port']}", 
        email=datasite['admin_email'],
        password=datasite['admin_password']
    )
```

#### **Pre-Experiment Connectivity Check**
```python
import subprocess
import sys

# Run connectivity test before experiments
result = subprocess.run([
    sys.executable, "test_datasite_connectivity.py", "--verbose"
], capture_output=True, text=True)

if result.returncode == 0:
    print("✅ All datasites accessible - proceeding with experiments")
    # Run federated learning experiments
else:
    print("❌ Datasite connectivity issues - please check configuration")
    print(result.stdout)
    sys.exit(1)
```

### Communication Manager Integration
```python
class PySyftCommunicationManager:
    def __init__(self, config_path: str = None):
        self.config_manager = DatasiteConfigManager(config_path)
        self.datasites = self.config_manager.get_datasite_configs()
        self.clients = {}
    
    def connect_to_datasites(self):
        """Establish connections to all configured datasites."""
        for datasite in self.datasites:
            try:
                client = sy.login(
                    url=f"http://{datasite['hostname']}:{datasite['port']}",
                    email=datasite['admin_email'],
                    password=datasite['admin_password']
                )
                self.clients[datasite['site_id']] = client
                print(f"✅ Connected to {datasite['site_id']}")
            except Exception as e:
                print(f"❌ Failed to connect to {datasite['site_id']}: {e}")
```

---

## 5. Production Deployment Guide

### Pre-Deployment Checklist

#### **Network Configuration**
- [ ] ✅ Update all hostnames from `localhost` to actual IP addresses
- [ ] ✅ Verify firewall rules allow datasite ports (8081, 8082, 8083)
- [ ] ✅ Configure Azure Network Security Groups for VM access
- [ ] ✅ Test network connectivity between coordinator and datasites

#### **Datasite Setup**
- [ ] ✅ PySyft datasites running on all target VMs
- [ ] ✅ Admin accounts created with correct credentials
- [ ] ✅ Datasite services configured to start automatically
- [ ] ✅ SSL certificates configured for production (optional)

#### **Configuration Validation**
```bash
# 1. Validate configuration file
python -c "from config.datasite_config import DatasiteConfigManager; print('✅ Valid' if DatasiteConfigManager().validate_config() else '❌ Invalid')"

# 2. Test all datasites
python test_datasite_connectivity.py --verbose

# 3. Expected output: All tests pass with ✅ status
```

### Security Considerations

#### **Credential Management**
```yaml
# Use environment variables for sensitive data
admin_email: ${DATASITE_ADMIN_EMAIL}
admin_password: ${DATASITE_ADMIN_PASSWORD}
```

#### **Network Security**
- Use private IP addresses for internal communication
- Configure VPN for cross-site connectivity
- Implement SSL/TLS for production deployments
- Restrict datasite access to authorized networks only

### Monitoring and Maintenance

#### **Automated Health Checks**
```bash
# Setup cron job for continuous monitoring
crontab -e
# Add: */15 * * * * /path/to/python test_datasite_connectivity.py >> /var/log/datasite_health.log
```

#### **Log Analysis**
```bash
# Monitor connectivity logs
tail -f /var/log/datasite_health.log

# Check for patterns
grep "❌" /var/log/datasite_health.log | tail -10
```

---

## 6. Troubleshooting Guide

### Common Issues and Solutions

#### **Connection Refused Errors**
```
❌ Network: Connection failed (error 111)
```
**Solutions:**
1. Verify datasite is running: `systemctl status datasite-service`
2. Check port availability: `netstat -tulpn | grep 8081`
3. Verify firewall rules: `ufw status`
4. Test from datasite VM: `curl localhost:8081`

#### **Name Resolution Failures**
```
❌ Network: Name resolution failed
```
**Solutions:**
1. Use IP addresses instead of hostnames
2. Add entries to `/etc/hosts` file
3. Configure DNS server for hostname resolution
4. Verify network connectivity: `ping hostname`

#### **Authentication Failures**
```
❌ PySyft: Authentication failed
```
**Solutions:**
1. Verify admin credentials in configuration
2. Check PySyft admin user exists on datasite
3. Reset admin password on datasite
4. Ensure datasite is fully initialized

#### **Configuration Loading Errors**
```
❌ Error loading config: File not found
```
**Solutions:**
1. Verify config file path: `ls -la config/datasite_configs.yaml`
2. Check file permissions: `chmod 644 config/datasite_configs.yaml`
3. Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('config/datasite_configs.yaml'))"`

### Diagnostic Commands

#### **Network Diagnostics**
```bash
# Test basic connectivity
telnet hostname port

# Check port status
nmap -p 8081-8083 hostname

# Trace network route
traceroute hostname
```

#### **PySyft Diagnostics**
```python
# Test PySyft installation
python -c "import syft; print(f'PySyft version: {syft.__version__}')"

# Test basic login
python -c "import syft as sy; client = sy.login(url='http://localhost:8081', email='admin@factory.com', password='password')"
```

---

## 7. Configuration Examples

### Development Environment
```yaml
# Local development with Docker containers
datasites:
  dev_site_1:
    site_id: dev_site_1
    hostname: localhost
    port: 8081
    admin_email: dev@localhost.com
    admin_password: devpass123
    site_name: Development_Site_1
```

### Staging Environment
```yaml
# Staging with VM deployment
datasites:
  staging_factory:
    site_id: staging_factory
    hostname: 10.0.1.100
    port: 8081
    admin_email: staging@company.com
    admin_password: staging_secure_pass
    site_name: Staging_Factory
```

### Production Environment
```yaml
# Production multi-site deployment
datasites:
  production_plant_east:
    site_id: production_plant_east
    hostname: 192.168.10.100
    port: 8081
    admin_email: admin@manufacturing.com
    admin_password: ${PROD_ADMIN_PASSWORD}
    site_name: East_Coast_Production_Plant
    
  production_plant_west:
    site_id: production_plant_west
    hostname: 192.168.20.100
    port: 8081
    admin_email: admin@manufacturing.com
    admin_password: ${PROD_ADMIN_PASSWORD}
    site_name: West_Coast_Production_Plant
```

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Framework Version**: 2.0  
**Configuration Format**: YAML/JSON  
**Testing Tool**: `test_datasite_connectivity.py`
