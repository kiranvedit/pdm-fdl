# Factory Datasite Windows Service

This directory contains files to run the Factory Datasite as a Windows Service with proper configuration management.

## Quick Start

### 1. Configure Your Factory
Choose your factory configuration:
```batch
# List available factory templates
python setup_factory_config.py --list

# Apply a specific factory configuration (e.g., SANFactory)
python setup_factory_config.py --factory SANFactory

# Or create custom configuration
python setup_factory_config.py --custom
```

### 2. Install the Service (Run as Administrator)
```batch
# Double-click or run as administrator:
install_service.bat
```

### 3. Manage the Service
```batch
# Start the service
service_manager.bat start

# Check status
service_manager.bat status

# Stop the service
service_manager.bat stop

# View logs
service_manager.bat logs
```

## Configuration Management

### Available Factory Templates
- **SANFactory** (172.21.0.4:8081)
- **SEAFactory** (172.17.0.4:8082)  
- **WUS2Factory** (172.19.0.4:8083)
- **SouthIndFactory** (172.18.0.4:8084)
- **NEFactory** (172.16.0.4:8085)

### Configuration Files
- **`service_config.json`** - Active service configuration
- **`config_templates/`** - Pre-configured factory templates
- **`setup_factory_config.py`** - Configuration management tool

### Example: Setup SANFactory
```batch
# Apply SANFactory configuration
python setup_factory_config.py --factory SANFactory

# Install and start service
install_service.bat
service_manager.bat start
```

## Files Description

- **`factory_datasite_service.py`** - Main Windows service wrapper
- **`service_config.json`** - Active service configuration
- **`setup_factory_config.py`** - Easy configuration management
- **`config_templates/`** - Pre-configured factory templates
- **`install_service.bat`** - One-click service installer (run as admin)
- **`uninstall_service.bat`** - Service uninstaller (run as admin)
- **`service_manager.bat`** - Service management commands

## Enhanced Features

### ✅ **Dynamic Configuration**
- No hardcoded values in heartbeat system
- Uses actual datasite hostname, port, and device information
- Configuration templates for different factory locations

### ✅ **Improved Heartbeat System**
- Sends heartbeats to both Dashboard and HeartbeatManager
- Uses actual datasite configuration (no hardcoded localhost:8080)
- Enhanced error handling and logging
- Better synchronization between systems

### ✅ **Easy Setup**
- Pre-configured templates for different factories
- Interactive custom configuration creation
- Simple factory switching with setup script

## Configuration Structure

```json
{
    "factory_name": "SANFactory",
    "port": 8081,
    "hostname": "172.21.0.4",
    "dashboard_url": "http://172.16.0.1:8888",
    "dev_mode": false,
    "reset": false,
    "verbose": true,
    "heartbeat_interval": 30
}
```

## Multi-Factory Deployment

### For Multiple Factories on Same Machine:
1. Create separate directories for each factory
2. Copy service files to each directory
3. Configure each with different factory template
4. Install each as separate Windows service

### For Network Deployment:
1. Use appropriate factory template for your location
2. Update hostname to actual machine IP
3. Ensure dashboard_url points to central orchestrator
4. Configure firewall for datasite port access

## Manual Installation

If you prefer manual installation:

1. Install dependencies:
   ```bash
   pip install pywin32
   ```

2. Install service:
   ```bash
   python factory_datasite_service.py install
   ```

3. Start service:
   ```bash
   python factory_datasite_service.py start
   ```

## Service Management Commands

```bash
# Install service
python factory_datasite_service.py install

# Remove service
python factory_datasite_service.py remove

# Start service
python factory_datasite_service.py start

# Stop service
python factory_datasite_service.py stop

# Restart service
python factory_datasite_service.py restart

# Check status
python factory_datasite_service.py status
```

## Alternative Windows Commands

You can also use standard Windows service commands:

```batch
# Start service
net start FactoryDatasiteService

# Stop service
net stop FactoryDatasiteService

# Check all services
sc query
```

## Logs

Service logs are stored in the `logs/` directory:
- `factory_datasite_service.log` - Service wrapper logs
- `factory_03.log` - Factory datasite application logs

## Troubleshooting

### Service Won't Start
1. Check logs: `service_manager.bat logs`
2. Verify Python path in `service_config.json`
3. Ensure all dependencies are installed
4. Check Windows Event Viewer for system errors

### Permission Issues
- Always run installation/removal as Administrator
- Ensure the service account has proper permissions

### Configuration Issues
- Verify `service_config.json` is valid JSON
- Check that ports are not in use by other applications
- Ensure dashboard URL is accessible

## Auto-Start on Boot

The service is configured to start automatically when Windows boots. To change this:

1. Open Windows Services manager (`services.msc`)
2. Find "Factory Datasite Service"
3. Right-click → Properties
4. Change "Startup type" as needed

## Security Considerations

- The service runs with LocalSystem privileges by default
- Consider creating a dedicated service account for production
- Ensure proper firewall configuration for the datasite port
- Review PySyft security settings for production deployment
