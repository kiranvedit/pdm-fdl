"""
Real-Time Datasite Status Monitoring and Diagnostics Guide
==========================================================

This guide shows you how to monitor and diagnose ALL connected datasites 
with TRUE status information, no fallbacks or hardcoded values.

OVERVIEW
--------
The enhanced federated learning system now includes comprehensive real-time 
monitoring that shows the exact status of every datasite based on actual 
heartbeat signals. The dashboard and diagnostic tools provide TRUE visibility 
into datasite availability and connectivity.

AVAILABLE TOOLS
--------------

1. STATUS DASHBOARD (Web Interface)
   - URL: http://localhost:8889
   - Shows real-time status of ALL registered datasites
   - Auto-refresh capability every 5 seconds
   - Force refresh button for immediate updates
   - Comprehensive diagnostics modal

2. DIAGNOSTIC METHODS (Python Commands)
   - show_diagnostics(): Comprehensive status analysis
   - show_datasite_status(): Quick status overview
   - force_refresh_status(): Force immediate status update
   - get_true_datasite_status(): Get raw status data
   - open_dashboard(): Open web dashboard

USAGE EXAMPLES
--------------

# Initialize experiment runner
runner = RealPySyftExperimentRunner()

# Show comprehensive diagnostics
runner.show_diagnostics()

# Show real-time datasite status
runner.show_datasite_status()

# Force immediate status refresh
runner.force_refresh_status()

# Get raw status data
status_data = runner.get_true_datasite_status()
print(json.dumps(status_data, indent=2))

# Open dashboard in browser
runner.open_dashboard()

DASHBOARD FEATURES
-----------------

Main Dashboard (http://localhost:8889):
- Real-time datasite status with visual indicators
- Online/offline status with last contact time
- Heartbeat count and source information
- Auto-refresh toggle (every 5 seconds)
- Force refresh button for immediate updates

Diagnostics Modal:
- Detailed status breakdown
- Never contacted datasites
- Status consistency checks
- Timestamp tracking

UNDERSTANDING STATUS INDICATORS
------------------------------

🟢 ONLINE (Green):
- Datasite has sent heartbeat within timeout period (90 seconds default)
- Available for training operations
- Shows actual connection status

🔴 OFFLINE (Red):
- No heartbeat received within timeout period
- Not available for training
- May indicate network issues or datasite shutdown

Status Information Displayed:
- Last Contact: Time since last heartbeat
- Heartbeats: Total number of heartbeats received
- Source: IP address and port of heartbeat sender
- Hostname: Datasite hostname/identifier

TROUBLESHOOTING COMMON ISSUES
-----------------------------

ISSUE: Dashboard shows no datasites
SOLUTION:
1. Check if heartbeat manager is running: runner.heartbeat_manager.running
2. Verify datasites are registered: runner.show_datasite_status()
3. Force refresh: runner.force_refresh_status()

ISSUE: Datasites showing as offline but should be online
SOLUTION:
1. Run diagnostics: runner.show_diagnostics()
2. Check for port conflicts (ensure start_heartbeat.py is NOT running)
3. Verify datasite heartbeat transmission
4. Force status refresh: runner.force_refresh_status()

ISSUE: Dashboard not accessible
SOLUTION:
1. Verify dashboard is running on port 8889
2. Check for port conflicts
3. Try opening manually: http://localhost:8889

ISSUE: Status not updating in real-time
SOLUTION:
1. Enable auto-refresh on dashboard
2. Use force refresh button
3. Check network connectivity

NO FALLBACK BEHAVIOR
--------------------

This system provides TRUE status only:
- No artificial "online" status for unavailable datasites
- No hardcoded fallback values
- Status reflects actual heartbeat timing
- Minimum 2 datasites required for experiments

Real-time status updates every:
- Heartbeat reception: Immediate
- Status cleanup: Every 30 seconds
- Dashboard sync: Every 60 seconds (or on-demand)
- Auto-refresh: Every 5 seconds (when enabled)

API ENDPOINTS
------------

The dashboard provides these API endpoints:

GET /api/status
- Returns current status of all datasites
- Includes summary statistics
- Forces real-time status update

GET /api/force_refresh
- Forces immediate refresh of all datasite statuses
- Returns sync results

GET /api/diagnostics
- Returns comprehensive diagnostic information
- Includes detailed status breakdown

INTEGRATION WITH EXPERIMENTS
---------------------------

During experiment execution:
1. Datasites are automatically registered with heartbeat manager
2. Status is monitored continuously
3. Experiments require minimum 2 online datasites
4. Dashboard shows training progress and datasite participation

For external datasites:
- Manual heartbeat refresh prevents timeout
- Real factory names used for identification
- Status reflects actual connectivity

For local datasites:
- Heartbeat signals from datasite processes
- Functional checks as backup verification
- Port-based identification

BEST PRACTICES
--------------

1. Always run diagnostics before starting experiments:
   runner.show_diagnostics()

2. Monitor dashboard during long experiments:
   http://localhost:8889

3. Force refresh if status seems incorrect:
   runner.force_refresh_status()

4. Check for port conflicts if issues persist:
   - Ensure start_heartbeat.py is NOT running
   - Verify ports 8888 and 8889 are available

5. Use real-time status for debugging:
   runner.show_datasite_status()

This monitoring system ensures you have complete visibility into the TRUE 
status of all connected datasites throughout your federated learning experiments.
"""
