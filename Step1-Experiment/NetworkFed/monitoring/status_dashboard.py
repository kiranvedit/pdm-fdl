"""
Simple Status Dashboard
======================
Basic web dashboard to monitor datasite status and experiment progress.
Only created if monitoring is needed.
"""

import json
import time
from datetime import datetime
from flask import Flask, render_template_string, jsonify
import threading
import logging

from monitoring.heartbeat_manager import get_heartbeat_manager


class SimpleStatusDashboard:
    """
    Simple web dashboard for monitoring federated learning experiments.
    """
    
    def __init__(self, port: int = 8889, heartbeat_manager=None):
        """
        Initialize status dashboard.
        
        Args:
            port: Port to run dashboard on
            heartbeat_manager: HeartbeatManager instance to use for datasite status
        """
        self.port = port
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)
        
        # Experiment tracking
        self.experiment_status = {
            'current_experiment': None,
            'current_round': 0,
            'total_experiments': 0,
            'completed_experiments': 0,
            'failed_experiments': 0,
            'start_time': None,
            'first_experiment_start_time': None  # Track when first experiment started
        }
        
        # Use provided heartbeat manager or get global instance
        self.heartbeat_manager = heartbeat_manager if heartbeat_manager else get_heartbeat_manager()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_HTML_TEMPLATE)
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for status data with real-time datasite information."""
            # Force real-time status update before returning data
            if hasattr(self.heartbeat_manager, 'force_sync_all_datasites'):
                self.heartbeat_manager.force_sync_all_datasites()
            
            datasite_status = self.heartbeat_manager.get_all_datasite_status()
            
            # DEBUG: Log datasite count information
            print(f"[DASHBOARD_DEBUG] Raw datasite_status from heartbeat manager: {len(datasite_status)} datasites")
            print(f"[DASHBOARD_DEBUG] Datasite names: {list(datasite_status.keys())}")
            
            # Ensure consistent timestamp formatting
            for datasite_id, status in datasite_status.items():
                if 'last_heartbeat' in status and status['last_heartbeat']:
                    # Convert datetime to ISO string for consistent JSON serialization
                    if hasattr(status['last_heartbeat'], 'isoformat'):
                        status['last_heartbeat'] = status['last_heartbeat'].isoformat()
            
            # Calculate real-time availability summary
            total_datasites = len(datasite_status)
            online_datasites = sum(1 for status in datasite_status.values() if status['status'] == 'online')
            
            # DEBUG: Log calculation details
            print(f"[DASHBOARD_DEBUG] Calculated total_datasites: {total_datasites}")
            print(f"[DASHBOARD_DEBUG] Calculated online_datasites: {online_datasites}")
            
            result = {
                'datasites': datasite_status,
                'datasite_summary': {
                    'total': total_datasites,
                    'online': online_datasites,
                    'offline': total_datasites - online_datasites
                },
                'experiment_status': self.experiment_status,
                'timestamp': datetime.now().isoformat()
            }
            
            # DEBUG: Log final result summary
            print(f"[DASHBOARD_DEBUG] API response summary - datasites: {len(result['datasites'])}, total: {result['datasite_summary']['total']}")
            
            return jsonify(result)
        
        @self.app.route('/api/force_refresh')
        def force_refresh():
            """Force immediate refresh of all datasite statuses."""
            if hasattr(self.heartbeat_manager, 'force_sync_all_datasites'):
                sync_results = self.heartbeat_manager.force_sync_all_datasites()
                return jsonify({
                    'success': True,
                    'message': 'Datasite status refreshed',
                    'sync_results': sync_results,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Force refresh not available',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/diagnostics')
        def diagnostics():
            """Get comprehensive diagnostics information."""
            if hasattr(self.heartbeat_manager, 'get_real_time_status_summary'):
                diagnostics_data = self.heartbeat_manager.get_real_time_status_summary()
                return jsonify(diagnostics_data)
            else:
                return jsonify({'error': 'Diagnostics not available'}), 500
    
    def start(self):
        """Start dashboard in background thread."""
        flask_thread = threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=False),
            daemon=True
        )
        flask_thread.start()
        print(f"📊 Status dashboard available at: http://localhost:{self.port}")
        time.sleep(1)  # Give Flask time to start
    
    def update_experiment_status(self, **kwargs):
        """Update experiment status."""
        self.experiment_status.update(kwargs)


# HTML template for the dashboard
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Federated Learning Status Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        
        /* Run indicator in main heading */
        .run-indicator {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin-left: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Enhanced Datasite Styling */
        .datasite-container { display: flex; flex-direction: column; gap: 10px; }
        .datasite { 
            display: flex; 
            align-items: center; 
            padding: 12px 15px; 
            border-radius: 8px; 
            margin: 8px 0; 
            border: 1px solid #ddd;
            transition: all 0.3s ease;
        }
        .datasite:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .datasite.online { background-color: #d4edda; border-left: 4px solid #28a745; border-color: #28a745; }
        .datasite.offline { background-color: #f8d7da; border-left: 4px solid #dc3545; border-color: #dc3545; }
        
        /* Status Indicator Dots */
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
            flex-shrink: 0;
            box-shadow: 0 0 0 2px white, 0 0 0 3px currentColor;
        }
        .status-dot.online { background-color: #28a745; color: #28a745; }
        .status-dot.offline { background-color: #dc3545; color: #dc3545; }
        .status-dot.warning { background-color: #ffc107; color: #ffc107; }
        
        /* Datasite Info Layout */
        .datasite-info { flex-grow: 1; }
        .datasite-name { font-weight: bold; font-size: 16px; margin-bottom: 4px; }
        .datasite-details { font-size: 12px; color: #666; line-height: 1.4; }
        .datasite-stats { 
            display: flex; 
            justify-content: space-between; 
            margin-top: 6px; 
            font-size: 11px; 
            color: #888; 
        }
        
        /* Summary Section */
        .datasite-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 10px;
        }
        .summary-item {
            text-align: center;
        }
        .summary-value {
            font-size: 24px;
            font-weight: bold;
            display: block;
        }
        .summary-label {
            font-size: 12px;
            opacity: 0.9;
        }
        
        .metric { display: inline-block; margin: 0 20px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 14px; color: #666; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
        .timestamp { color: #666; font-size: 12px; }
        
        /* Experiment Status Cards */
        .experiment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .experiment-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .experiment-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(0,0,0,0.15);
        }
        
        .experiment-card.current { border-left-color: #28a745; }
        .experiment-card.progress { border-left-color: #ffc107; }
        .experiment-card.performance { border-left-color: #17a2b8; }
        
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .card-icon {
            font-size: 24px;
            margin-right: 10px;
        }
        
        .card-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        
        /* Progress Ring */
        .progress-ring {
            position: relative;
            width: 80px;
            height: 80px;
            margin: 10px auto;
        }
        
        .progress-ring svg {
            transform: rotate(-90deg);
            width: 100%;
            height: 100%;
        }
        
        .progress-ring circle {
            fill: none;
            stroke-width: 6;
        }
        
        .progress-ring .bg {
            stroke: #e9ecef;
        }
        
        .progress-ring .progress {
            stroke: #28a745;
            stroke-linecap: round;
            transition: stroke-dasharray 0.3s ease;
        }
        
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 14px;
            color: #333;
        }
        
        /* Progress Bar */
        .progress-bar-container {
            background: #e9ecef;
            border-radius: 10px;
            height: 8px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }
        
        .status-badge.running { background: #d4edda; color: #155724; }
        .status-badge.completed { background: #cce5ff; color: #004085; }
        .status-badge.failed { background: #f8d7da; color: #721c24; }
        .status-badge.warning { background: #fff3cd; color: #856404; }
        
        /* Card metrics */
        .card-metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
            padding: 8px 0;
            border-bottom: 1px solid #f1f1f1;
        }
        
        .card-metric:last-child {
            border-bottom: none;
        }
        
        .metric-label-small {
            font-size: 13px;
            color: #666;
        }
        
        .metric-value-small {
            font-weight: bold;
            font-size: 14px;
            color: #333;
        }
        
        /* Pulse animation for active datasites */
        .status-dot.online {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 2px white, 0 0 0 3px #28a745; }
            70% { box-shadow: 0 0 0 2px white, 0 0 0 8px rgba(40, 167, 69, 0.4); }
            100% { box-shadow: 0 0 0 2px white, 0 0 0 3px #28a745; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1 class="header">🔗 Federated Learning Status Dashboard</h1>
            <div style="text-align: center;">
                <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
                <button class="refresh-btn" onclick="forceRefresh()" style="background: #28a745; margin-left: 10px;">⚡ Force Refresh</button>
                <button class="refresh-btn" onclick="showDiagnostics()" style="background: #17a2b8; margin-left: 10px;">🔍 Diagnostics</button>
                <span class="timestamp" id="lastUpdate"></span>
            </div>
        </div>

        <div class="status-grid">
            <div class="card">
                <h2>📊 Experiment Status <span id="runIndicator" class="run-indicator">Run 1/1</span></h2>
                <div id="experimentStatus">Loading...</div>
            </div>

            <div class="card">
                <h2>🏭 Datasite Status 
                    <button onclick="toggleAutoRefresh()" id="autoRefreshBtn" style="background: #6c757d; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;">
                        Auto: OFF
                    </button>
                </h2>
                <div id="datasiteStatus">Loading...</div>
            </div>
        </div>

        <div class="card">
            <h2>📈 System Metrics</h2>
            <div id="systemMetrics">Loading...</div>
        </div>
        
        <!-- Diagnostics Modal -->
        <div id="diagnosticsModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000;">
            <div style="position: relative; margin: 5% auto; width: 80%; max-width: 800px; background: white; border-radius: 8px; padding: 20px; max-height: 80%; overflow-y: auto;">
                <span onclick="closeDiagnostics()" style="position: absolute; top: 10px; right: 15px; font-size: 28px; cursor: pointer;">&times;</span>
                <h2>🔍 Real-Time Diagnostics</h2>
                <div id="diagnosticsContent">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        let autoRefreshInterval = null;
        let autoRefreshEnabled = false;
        
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateExperimentStatus(data.experiment_status);
                    updateDatasiteStatus(data.datasites, data.datasite_summary);
                    updateSystemMetrics(data);
                    document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date(data.timestamp).toLocaleString();
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('lastUpdate').textContent = 'Update failed: ' + new Date().toLocaleString();
                });
        }
        
        function forceRefresh() {
            fetch('/api/force_refresh')
                .then(response => response.json())
                .then(result => {
                    if (result.success) {
                        console.log('Force refresh successful:', result.sync_results);
                        refreshData(); // Immediately refresh display
                    } else {
                        console.error('Force refresh failed:', result.error);
                    }
                })
                .catch(error => {
                    console.error('Force refresh error:', error);
                });
        }
        
        function showDiagnostics() {
            fetch('/api/diagnostics')
                .then(response => response.json())
                .then(data => {
                    updateDiagnosticsDisplay(data);
                    document.getElementById('diagnosticsModal').style.display = 'block';
                })
                .catch(error => {
                    console.error('Diagnostics error:', error);
                });
        }
        
        function closeDiagnostics() {
            document.getElementById('diagnosticsModal').style.display = 'none';
        }
        
        function toggleAutoRefresh() {
            const btn = document.getElementById('autoRefreshBtn');
            
            if (autoRefreshEnabled) {
                // Disable auto refresh
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
                autoRefreshEnabled = false;
                btn.textContent = 'Auto: OFF';
                btn.style.background = '#6c757d';
            } else {
                // Enable auto refresh every 5 seconds
                autoRefreshInterval = setInterval(refreshData, 5000);
                autoRefreshEnabled = true;
                btn.textContent = 'Auto: ON';
                btn.style.background = '#28a745';
            }
        }
        
        function updateDiagnosticsDisplay(data) {
            let html = `
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h3>📊 Status Summary</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center;">
                        <div>
                            <div style="font-size: 24px; font-weight: bold; color: #333;">${data.total_registered}</div>
                            <div style="font-size: 12px; color: #666;">Total Registered</div>
                        </div>
                        <div>
                            <div style="font-size: 24px; font-weight: bold; color: #28a745;">${data.online_count}</div>
                            <div style="font-size: 12px; color: #666;">Online</div>
                        </div>
                        <div>
                            <div style="font-size: 24px; font-weight: bold; color: #dc3545;">${data.offline_count}</div>
                            <div style="font-size: 12px; color: #666;">Offline</div>
                        </div>
                        <div>
                            <div style="font-size: 24px; font-weight: bold; color: #ffc107;">${data.never_contacted_count}</div>
                            <div style="font-size: 12px; color: #666;">Never Contacted</div>
                        </div>
                    </div>
                </div>
                
                <h3>🏭 Detailed Datasite Status</h3>
                <div style="max-height: 400px; overflow-y: auto;">
            `;
            
            for (const [name, info] of Object.entries(data.datasites)) {
                const statusIcon = info.status === 'online' ? '🟢' : '🔴';
                const lastContact = info.seconds_since_heartbeat ? `${info.seconds_since_heartbeat.toFixed(1)}s ago` : 'Never';
                
                html += `
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 8px; background: ${info.status === 'online' ? '#d4edda' : '#f8d7da'};">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 16px; margin-right: 8px;">${statusIcon}</span>
                            <strong>${name}</strong>
                            <span style="margin-left: auto; font-size: 12px; color: #666;">${info.status.toUpperCase()}</span>
                        </div>
                        <div style="font-size: 12px; color: #666; line-height: 1.4;">
                            <div>Last Contact: ${lastContact}</div>
                            <div>Heartbeats: ${info.heartbeat_count}</div>
                            <div>Source: ${info.client_ip}:${info.port}</div>
                            <div>Hostname: ${info.hostname}</div>
                            <div>Within Timeout: ${info.within_timeout ? '✅ Yes' : '❌ No'}</div>
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            html += `<div style="text-align: center; margin-top: 15px; font-size: 12px; color: #666;">Last Updated: ${new Date(data.timestamp).toLocaleString()}</div>`;
            
            document.getElementById('diagnosticsContent').innerHTML = html;
        }

        function updateExperimentStatus(status) {
            // Update run indicator in the main heading
            const runIndicator = document.getElementById('runIndicator');
            if (runIndicator && status.current_run) {
                const totalRuns = status.total_runs || '?';
                runIndicator.textContent = `Run ${status.current_run}/${totalRuns}`;
            }
            
            // Calculate progress metrics
            const roundProgress = status.total_rounds ? (status.current_round / status.total_rounds * 100) : 0;
            
            // Determine if we're showing current run or overall progress
            // Show "Current Run Progress" only if we're in a multi-run scenario AND have experiments per run data
            // Show "Overall Progress" when resuming or showing total progress across all experiments
            const hasMultipleRuns = status.total_runs && status.total_runs > 1;
            const hasRunSpecificData = status.experiments_per_run && status.completed_experiments_current_run !== undefined;
            const isActivelyRunning = status.current_experiment && status.current_round > 0;
            
            // Use "Current Run Progress" only when we're actively in a multi-run scenario with run-specific tracking
            const isCurrentRunProgress = hasMultipleRuns && hasRunSpecificData && isActivelyRunning;
            const progressTitle = isCurrentRunProgress ? "Current Run Progress" : "Overall Progress";
            
            // Calculate progress based on whether we're showing current run or overall
            let currentRunCompleted, currentRunTotal, overallProgress;
            
            if (isCurrentRunProgress && status.experiments_per_run) {
                // Show current run progress
                currentRunCompleted = status.completed_experiments_current_run || 0;
                currentRunTotal = status.experiments_per_run;
                overallProgress = currentRunTotal > 0 ? (currentRunCompleted / currentRunTotal * 100) : 0;
            } else {
                // Show overall progress (default for resuming and single runs)
                currentRunCompleted = status.completed_experiments || 0;
                currentRunTotal = status.total_experiments || 48;
                overallProgress = currentRunTotal > 0 ? (currentRunCompleted / currentRunTotal * 100) : 0;
                
                // IMPORTANT: Prevent reset when resuming
                // If we have a reasonable completed count, don't let it drop to a very low number unexpectedly
                if (window.lastKnownCompleted && currentRunCompleted < window.lastKnownCompleted && 
                    currentRunCompleted < 5 && window.lastKnownCompleted > 10) {
                    console.warn('[DASHBOARD_DEBUG] Potential progress reset detected!', {
                        lastKnownCompleted: window.lastKnownCompleted,
                        newCompleted: currentRunCompleted,
                        'Keeping previous value': true
                    });
                    // Keep the last known higher value to prevent unexpected resets
                    currentRunCompleted = window.lastKnownCompleted;
                    overallProgress = currentRunTotal > 0 ? (currentRunCompleted / currentRunTotal * 100) : 0;
                }
                
                // Remember the completed count for reset detection
                if (currentRunCompleted > 0) {
                    window.lastKnownCompleted = currentRunCompleted;
                }
            }
            
            // DEBUG: Log progress calculation details
            console.log('[DASHBOARD_DEBUG] Progress calculation:', {
                progressTitle: progressTitle,
                isCurrentRunProgress: isCurrentRunProgress,
                hasMultipleRuns: hasMultipleRuns,
                hasRunSpecificData: hasRunSpecificData,
                isActivelyRunning: isActivelyRunning,
                currentRunCompleted: currentRunCompleted,
                currentRunTotal: currentRunTotal,
                overallProgress: overallProgress,
                'status.completed_experiments': status.completed_experiments,
                'status.completed_experiments_current_run': status.completed_experiments_current_run,
                'status.total_experiments': status.total_experiments,
                'status.experiments_per_run': status.experiments_per_run
            });
            
            const successRate = currentRunCompleted > 0 ? 
                ((currentRunCompleted - (status.failed_experiments || 0)) / currentRunCompleted * 100) : 100;
            
            // Calculate experiment duration
            let experimentDuration = 'Unknown';
            let avgTimePerRound = 'Calculating...';
            if (status.start_time) {
                const startTime = new Date(status.start_time);
                const now = new Date();
                const durationMs = now - startTime;
                const durationMinutes = Math.floor(durationMs / (1000 * 60));
                
                if (durationMinutes < 60) {
                    experimentDuration = `${durationMinutes}m`;
                } else {
                    const hours = Math.floor(durationMinutes / 60);
                    const minutes = durationMinutes % 60;
                    experimentDuration = `${hours}h ${minutes}m`;
                }
                
                // Calculate average time per round (if we have completed any)
                if (status.current_round > 0) {
                    const avgMs = durationMs / status.current_round;
                    const avgMinutes = Math.floor(avgMs / (1000 * 60));
                    const avgSeconds = Math.floor((avgMs % (1000 * 60)) / 1000);
                    avgTimePerRound = avgMinutes > 0 ? `${avgMinutes}m ${avgSeconds}s` : `${avgSeconds}s`;
                }
            }
            
            // Calculate estimated completion time
            let eta = 'Calculating...';
            if (status.current_round > 0 && status.total_rounds && status.start_time) {
                const startTime = new Date(status.start_time);
                const now = new Date();
                const elapsedMs = now - startTime;
                const avgTimePerRound = elapsedMs / status.current_round;
                const remainingRounds = status.total_rounds - status.current_round;
                const etaMs = remainingRounds * avgTimePerRound;
                const etaMinutes = Math.floor(etaMs / (1000 * 60));
                
                if (etaMinutes < 60) {
                    eta = `${etaMinutes}m`;
                } else {
                    const hours = Math.floor(etaMinutes / 60);
                    const minutes = etaMinutes % 60;
                    eta = `${hours}h ${minutes}m`;
                }
            }
            
            // Determine experiment status
            let experimentStatus = 'idle';
            let statusBadge = 'idle';
            if (status.current_experiment && status.current_round > 0) {
                experimentStatus = 'running';
                statusBadge = 'running';
            } else if (currentRunCompleted === currentRunTotal) {
                experimentStatus = 'completed';
                statusBadge = 'completed';
            }
            
            const html = `
                <div class="experiment-grid">
                    <!-- Current Experiment Card -->
                    <div class="experiment-card current">
                        <div class="card-header">
                            <span class="card-icon">🔬</span>
                            <span class="card-title">Current Experiment</span>
                        </div>
                        
                        <div style="text-align: center;">
                            <div class="progress-ring">
                                <svg>
                                    <circle class="bg" cx="40" cy="40" r="30"></circle>
                                    <circle class="progress" cx="40" cy="40" r="30" 
                                            stroke-dasharray="${roundProgress * 1.88} 188.5"></circle>
                                </svg>
                                <div class="progress-text">${Math.round(roundProgress)}%</div>
                            </div>
                            
                            <div style="margin-top: 10px;">
                                <div style="font-weight: bold; font-size: 16px; margin-bottom: 5px;">
                                    ${status.current_experiment || 'No active experiment'}
                                </div>
                                <span class="status-badge ${statusBadge}">
                                    ${experimentStatus.toUpperCase()}
                                </span>
                            </div>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Round Progress</span>
                            <span class="metric-value-small">${status.current_round || 0}/${status.total_rounds || '?'}</span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Duration</span>
                            <span class="metric-value-small">${experimentDuration}</span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">ETA</span>
                            <span class="metric-value-small">${eta}</span>
                        </div>
                    </div>
                    
                    <!-- Overall/Current Run Progress Card -->
                    <div class="experiment-card progress">
                        <div class="card-header">
                            <span class="card-icon">📊</span>
                            <span class="card-title">${progressTitle}</span>
                        </div>
                        
                        <div style="text-align: center; margin-bottom: 15px;">
                            <div style="font-size: 24px; font-weight: bold; color: #333;">
                                ${currentRunCompleted}/${currentRunTotal}
                            </div>
                            <div style="font-size: 14px; color: #666;">
                                ${isCurrentRunProgress ? 'Current Run Experiments' : 'Experiments Completed'}
                            </div>
                        </div>
                        
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: ${overallProgress}%"></div>
                        </div>
                        <div style="text-align: center; font-size: 12px; color: #666; margin-bottom: 15px;">
                            ${Math.round(overallProgress)}% Complete
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Completed</span>
                            <span class="metric-value-small" style="color: #28a745;">${currentRunCompleted}</span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Failed</span>
                            <span class="metric-value-small" style="color: #dc3545;">${status.failed_experiments || 0}</span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Remaining</span>
                            <span class="metric-value-small" style="color: #6c757d;">
                                ${currentRunTotal - currentRunCompleted}
                            </span>
                        </div>
                    </div>
                    
                    <!-- Performance Card -->
                    <div class="experiment-card performance">
                        <div class="card-header">
                            <span class="card-icon">⚡</span>
                            <span class="card-title">Performance Metrics</span>
                        </div>
                        
                        <div style="text-align: center; margin-bottom: 15px;">
                            <div style="font-size: 24px; font-weight: bold; color: #17a2b8;">
                                ${Math.round(successRate)}%
                            </div>
                            <div style="font-size: 14px; color: #666;">Success Rate</div>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Current Run</span>
                            <span class="metric-value-small" style="color: #007bff; font-weight: bold;">
                                ${status.current_run || 1}${status.total_runs ? `/${status.total_runs}` : ''}
                            </span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Avg Time/Round</span>
                            <span class="metric-value-small">${avgTimePerRound}</span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Session Duration</span>
                            <span class="metric-value-small">${experimentDuration}</span>
                        </div>
                        
                        <div class="card-metric">
                            <span class="metric-label-small">Health Status</span>
                            <span class="metric-value-small">
                                ${(status.failed_experiments || 0) === 0 ? 
                                    '<span style="color: #28a745;">●</span> Excellent' : 
                                    (status.failed_experiments || 0) < 3 ? 
                                        '<span style="color: #ffc107;">●</span> Good' : 
                                        '<span style="color: #dc3545;">●</span> Needs Attention'
                                }
                            </span>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('experimentStatus').innerHTML = html;
        }

        function updateDatasiteStatus(datasites, summary) {
            let html = `
                <div class="datasite-summary">
                    <h3>Network Overview</h3>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <span class="summary-value">${summary.total}</span>
                            <span class="summary-label">Total</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-value" style="color: #4CAF50;">${summary.online}</span>
                            <span class="summary-label">Online</span>
                        </div>
                        <div class="summary-item">
                            <span class="summary-value" style="color: #f44336;">${summary.total - summary.online}</span>
                            <span class="summary-label">Offline</span>
                        </div>
                    </div>
                </div>
                <div class="datasite-container">
            `;
            
            // Sort datasites for consistent display
            const sortedDatasites = Object.entries(datasites).sort(([a], [b]) => a.localeCompare(b));
            
            for (const [name, status] of sortedDatasites) {
                try {
                    const isOnline = status.status === 'online';
                    const statusClass = isOnline ? 'online' : 'offline';
                    const statusDotClass = isOnline ? 'online' : 'offline';
                    
                    // Safe timestamp handling to prevent JavaScript errors from skipping datasites
                    let lastHeartbeatTime = 'Unknown';
                    let uptimeDisplay = 'Unknown';
                    
                    try {
                        if (status.last_heartbeat) {
                            const lastHeartbeat = new Date(status.last_heartbeat);
                            // Check if date is valid (not NaN)
                            if (!isNaN(lastHeartbeat.getTime())) {
                                lastHeartbeatTime = lastHeartbeat.toLocaleTimeString();
                                
                                // Calculate time since last heartbeat
                                const now = new Date();
                                const diffMinutes = Math.floor((now - lastHeartbeat) / (1000 * 60));
                                
                                if (isOnline) {
                                    uptimeDisplay = 'Active now';
                                } else if (diffMinutes < 60) {
                                    uptimeDisplay = `${diffMinutes}m ago`;
                                } else if (diffMinutes < 1440) {
                                    uptimeDisplay = `${Math.floor(diffMinutes/60)}h ago`;
                                } else {
                                    uptimeDisplay = `${Math.floor(diffMinutes/1440)}d ago`;
                                }
                            } else {
                                console.warn(`Invalid timestamp for datasite ${name}: ${status.last_heartbeat}`);
                                lastHeartbeatTime = 'Invalid';
                                uptimeDisplay = 'Unknown';
                            }
                        } else {
                            lastHeartbeatTime = 'Never';
                            uptimeDisplay = 'Never contacted';
                        }
                    } catch (timeError) {
                        console.error(`Timestamp parsing error for datasite ${name}:`, timeError);
                        lastHeartbeatTime = 'Error';
                        uptimeDisplay = 'Parse error';
                    }
                    
                    html += `
                        <div class="datasite ${statusClass}">
                            <div class="status-dot ${statusDotClass}"></div>
                            <div class="datasite-info">
                                <div class="datasite-name">${name}</div>
                                <div class="datasite-details">
                                    <div>Status: ${isOnline ? 'Connected' : 'Disconnected'}</div>
                                    <div>Last seen: ${uptimeDisplay}</div>
                                </div>
                                <div class="datasite-stats">
                                    <span>Heartbeats: ${status.heartbeat_count || 0}</span>
                                    <span>Time: ${lastHeartbeatTime}</span>
                                </div>
                            </div>
                        </div>
                    `;
                } catch (datasiteError) {
                    console.error(`Error rendering datasite ${name}:`, datasiteError);
                    // Still include the datasite even if there's an error
                    html += `
                        <div class="datasite offline">
                            <div class="status-dot offline"></div>
                            <div class="datasite-info">
                                <div class="datasite-name">${name}</div>
                                <div class="datasite-details">
                                    <div>Status: Error</div>
                                    <div>Last seen: Render error</div>
                                </div>
                                <div class="datasite-stats">
                                    <span>Heartbeats: ${status.heartbeat_count || 0}</span>
                                    <span>Time: Error</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
            }
            
            html += '</div>';
            document.getElementById('datasiteStatus').innerHTML = html;
        }

        function updateSystemMetrics(data) {
            const startTime = data.experiment_status.start_time ? new Date(data.experiment_status.start_time).toLocaleString() : 'Not started';
            const firstExperimentStartTime = data.experiment_status.first_experiment_start_time ? new Date(data.experiment_status.first_experiment_start_time).toLocaleString() : 'Not started';
            
            // Calculate total elapsed time since first experiment started
            let totalElapsedTime = 'Not started';
            if (data.experiment_status.first_experiment_start_time) {
                const firstStartTime = new Date(data.experiment_status.first_experiment_start_time);
                const now = new Date();
                const elapsedMs = now - firstStartTime;
                const elapsedHours = Math.floor(elapsedMs / (1000 * 60 * 60));
                const elapsedMinutes = Math.floor((elapsedMs % (1000 * 60 * 60)) / (1000 * 60));
                
                if (elapsedHours > 0) {
                    totalElapsedTime = `${elapsedHours}h ${elapsedMinutes}m`;
                } else {
                    totalElapsedTime = `${elapsedMinutes}m`;
                }
            }
            
            const html = `
                <div class="metric">
                    <div class="metric-value">${data.datasite_summary.online}</div>
                    <div class="metric-label">Available Datasites</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${firstExperimentStartTime}</div>
                    <div class="metric-label">First Experiment Started</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${totalElapsedTime}</div>
                    <div class="metric-label">Total Elapsed Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${startTime}</div>
                    <div class="metric-label">Current Experiment Start</div>
                </div>
            `;
            document.getElementById('systemMetrics').innerHTML = html;
        }

        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
"""


# Global dashboard instance
_dashboard = None

def start_status_dashboard(port: int = 8889):
    """Start the status dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = SimpleStatusDashboard(port)
        _dashboard.start()
    return _dashboard

def get_dashboard():
    """Get the dashboard instance."""
    return _dashboard
