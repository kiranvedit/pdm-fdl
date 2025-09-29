# Heartbeat Manager Usage Guide

## ⚠️ IMPORTANT: Do NOT Run start_heartbeat.py Manually

### The Problem
There are **TWO** heartbeat managers in the codebase:
1. **Standalone heartbeat manager** (`start_heartbeat.py`) - Independent service
2. **Integrated heartbeat manager** (in `run_enhanced_experiments.py`) - Built into experiment runner

Running both causes **PORT CONFLICTS** on port 8888!

## ✅ Correct Usage

### For Experiments:
- **DO NOT** run `start_heartbeat.py` before starting experiments
- The experiment runner **automatically starts its own heartbeat manager**
- Port 8888 is used by the integrated heartbeat manager
- Port 8889 is used by the status dashboard

### For Standalone Testing:
- **ONLY** use `start_heartbeat.py` when testing heartbeat functionality without experiments
- Stop `start_heartbeat.py` before running any experiments

## 🔍 Troubleshooting

### Symptoms of Port Conflict:
- Datasites showing as "Disconnected" despite sending heartbeats
- Inconsistent datasite availability detection
- Experiments hanging during datasite availability checks

### How to Fix:
1. **Stop any running start_heartbeat.py processes**
   ```bash
   # Check for running processes
   tasklist | findstr python
   # Kill if found
   taskkill /F /PID <process_id>
   ```

2. **Check port 8888 availability**
   ```bash
   netstat -an | findstr :8888
   ```

3. **Start experiments normally** - the integrated heartbeat manager will start automatically

### Verification:
The experiment runner now includes automatic port conflict detection:
- ✅ Shows "Port 8888 is available" if no conflicts
- ⚠️ Shows warning if another heartbeat manager is detected

## 🚀 Recent Improvements

### Enhanced Startup Process:
1. **Port Conflict Detection**: Checks if port 8888 is already in use
2. **Extended Startup Time**: 5-second stabilization period
3. **Startup Verification**: Confirms heartbeat manager is responding
4. **Better Error Messages**: Clear indication of port conflicts

### Robust Heartbeat Processing:
1. **Improved JSON Validation**: Better error handling for malformed heartbeats
2. **Enhanced Logging**: Detailed status transitions and heartbeat reception
3. **Comprehensive Diagnostics**: Built-in diagnostic tools for debugging

### Unified Availability Logic:
1. **Consistent for All Datasites**: Both local and external datasites use heartbeat + functional fallback
2. **Detailed Logging**: Clear indication of which availability method was used
3. **Status Synchronization**: Updates heartbeat status when functional check succeeds

## 📊 Diagnostic Commands

### In Python/Experiment Code:
```python
# Print comprehensive heartbeat status
if hasattr(runner, 'heartbeat_manager') and runner.heartbeat_manager:
    runner.heartbeat_manager.print_diagnostic_report()
```

### API Endpoints (when heartbeat manager is running):
- `GET http://localhost:8888/status` - All datasite status
- `POST http://localhost:8888/heartbeat` - Receive heartbeat (for datasites)

## 🎯 Best Practices

1. **Never run start_heartbeat.py with experiments**
2. **Check for port conflicts** if datasites appear disconnected
3. **Let the experiment runner manage heartbeats** automatically
4. **Use diagnostic tools** to debug availability issues
5. **Ensure external datasites send heartbeats to port 8888** (not 8889)

---

**Remember**: The integrated heartbeat manager in the experiment runner is the preferred and most robust solution for production experiments.
