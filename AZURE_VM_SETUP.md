# Azure VM Environment Setup Guide

## Python Version
**Exact Version Required: Python 3.12.10**

## System Requirements

### For Ubuntu/Debian Azure VMs:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12.10
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# For CUDA support (if using GPU VMs)
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit -y
```

### For Windows Azure VMs:
```powershell
# Download and install Python 3.12.10 from python.org
# OR use chocolatey:
choco install python --version=3.12.10

# Install Visual C++ Build Tools (required for some packages)
choco install visualstudio2022buildtools
```

## Environment Setup

### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd /path/to/pdm-fdl

# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. Upgrade pip and install requirements
```bash
# Upgrade pip to latest version
pip install --upgrade pip==25.2

# Install all requirements
pip install -r azure_vm_requirements.txt
```

### 3. Verify Installation
```python
# Run this Python script to verify installation
import sys
print(f"Python version: {sys.version}")

# Test key packages
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    import syft
    print(f"PySyft version: {syft.__version__}")
    
    import flwr
    print(f"Flower version: {flwr.__version__}")
    
    import numpy as np
    import pandas as pd
    import sklearn
    print("All core packages imported successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
```

## Network Configuration for Distributed Experiments

### Firewall Ports (Azure NSG Rules)
- **8080-8090**: PySyft datasite ports
- **8765-8775**: WebSocket communication
- **5000-5010**: Flask/FastAPI services
- **6443**: Kubernetes API (if using k8s)
- **22**: SSH access
- **443/80**: HTTPS/HTTP

### Azure VM Recommendations
- **VM Size**: Standard_NC6s_v3 or higher (for GPU support)
- **OS Disk**: Premium SSD (100GB minimum)
- **Data Disk**: Premium SSD (500GB for datasets)
- **Network**: Accelerated networking enabled
- **Region**: Choose regions close to your datasites for low latency

### Environment Variables
```bash
# Add to ~/.bashrc or VM startup script
export PYTHONPATH="/path/to/pdm-fdl:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export SYFT_DATASITE_PORT=8080
export SYFT_NETWORK_URL="https://your-network-url"
```

## Distributed Deployment Checklist

1. ✅ Install Python 3.12.10
2. ✅ Install all requirements from azure_vm_requirements.txt
3. ✅ Configure firewall/NSG rules
4. ✅ Set environment variables
5. ✅ Test PySyft datasite connectivity
6. ✅ Verify CUDA support (for GPU VMs)
7. ✅ Clone pdm-fdl repository
8. ✅ Run connectivity tests between VMs

## Troubleshooting

### Common Issues:
1. **CUDA not detected**: Ensure NVIDIA drivers are installed
2. **Package conflicts**: Use exact versions from requirements file
3. **Network connectivity**: Check Azure NSG and VM firewall rules
4. **Memory issues**: Increase VM size or add swap space
5. **PySyft connection errors**: Verify ports and SSL certificates

### Performance Optimization:
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize network settings
echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf
sysctl -p
```
