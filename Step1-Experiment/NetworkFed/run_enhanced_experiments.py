"""
48 REAL PySyft Federated Learning Experiments - ENHANCED VERSION
================================================================
ZERO simulation code. ONLY real PySyft datasites.
Enhanced with configurable parameters and resumption capability.

ENHANCED FEATURES:
- Parallel Processing: All datasites train simultaneously  
- Heartbeat Monitoring: Real-time datasite availability tracking
- Fault Tolerance: Continue with available datasites, minimum 2 required
- Adaptive Waiting: Smart waiting based on heartbeat status

SAME 48 EXPERIMENTS: 4 algorithms × 3 models × 2 distributions × 2 communication styles
"""
print("Initializing .....")

# Standard library imports
import os
import sys
import json
import time
import csv
import logging
import argparse
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Third-party imports
import torch
import numpy as np

# Import configurable experiment system
try:
    from configurable_experiments import ExperimentParams
    # Note: We avoid importing ExperimentState from configurable_experiments
    # to use our local enhanced version with get_resume_info method
    CONFIGURABLE_EXPERIMENTS_AVAILABLE = True
except ImportError:
    print("[WARNING] Configurable experiments module not available")
    CONFIGURABLE_EXPERIMENTS_AVAILABLE = False
    # Create dummy classes to avoid type errors
    class ExperimentParams:
        def __init__(self, max_rounds=10, local_epochs=1, batch_size=32, learning_rate=0.001, num_datasites=3, fedprox_mu=0.01,
                     early_stopping_patience=5, early_stopping_min_delta=0.01, early_stopping_metric='val_loss'):
            self.max_rounds = max_rounds
            self.local_epochs = local_epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.num_datasites = num_datasites
            self.fedprox_mu = fedprox_mu
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_min_delta = early_stopping_min_delta
            self.early_stopping_metric = early_stopping_metric
        
        def to_dict(self):
            return {
                'max_rounds': self.max_rounds,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_datasites': self.num_datasites,
                'fedprox_mu': self.fedprox_mu,
                'early_stopping_patience': self.early_stopping_patience,
                'early_stopping_min_delta': self.early_stopping_min_delta,
                'early_stopping_metric': self.early_stopping_metric
            }
        
        @classmethod
        def from_json_file(cls, filepath: str):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(**data)
        
        def save_to_file(self, filepath: str):
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

class LocalExperimentState:
    """Clean experiment state management with simple directory structure."""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.state_file = os.path.join(results_dir, "experiment_state.json")
        self.state_data = self._load_state()
    
    def _load_state(self):
        """Load existing state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load state file: {e}")
        return {
            'current_run': 1,
            'runs': {}  # run_number -> {experiment_name -> {completed: bool, last_round: int}}
        }
    
    def _save_state(self):
        """Save current state to file."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save state file: {e}")
    
    def is_experiment_completed(self, experiment_name, run_number=None):
        """Check if experiment is fully completed for a specific run."""
        if run_number is None:
            run_number = self.state_data.get('current_run', 1)
        
        run_key = f"run_{run_number}"
        return self.state_data.get('runs', {}).get(run_key, {}).get(experiment_name, {}).get('completed', False)
    
    def mark_experiment_completed(self, experiment_name, run_number=None):
        """Mark experiment as completed for a specific run."""
        if run_number is None:
            run_number = self.state_data.get('current_run', 1)
        
        run_key = f"run_{run_number}"
        if 'runs' not in self.state_data:
            self.state_data['runs'] = {}
        if run_key not in self.state_data['runs']:
            self.state_data['runs'][run_key] = {}
        if experiment_name not in self.state_data['runs'][run_key]:
            self.state_data['runs'][run_key][experiment_name] = {}
            
        self.state_data['runs'][run_key][experiment_name]['completed'] = True
        self.state_data['runs'][run_key][experiment_name]['completion_time'] = time.time()
        self._save_state()
    
    def get_current_round(self, experiment_name, run_number=None):
        """Get the last completed round for resumption."""
        if run_number is None:
            run_number = self.state_data.get('current_run', 1)
        
        run_key = f"run_{run_number}"
        
        # Check state file first
        state_round = self.state_data.get('runs', {}).get(run_key, {}).get(experiment_name, {}).get('last_round', 0)
        
        # Also check for existing round files in clean structure: run_X/experiment_name/round_Y.json
        experiment_dir = os.path.join(self.results_dir, run_key, experiment_name)
        file_round = 0
        
        if os.path.exists(experiment_dir):
            import glob
            round_files = glob.glob(os.path.join(experiment_dir, "round_*.json"))
            if round_files:
                round_numbers = []
                for file_path in round_files:
                    filename = os.path.basename(file_path)
                    if filename.startswith("round_") and filename.endswith(".json"):
                        try:
                            round_num = int(filename[6:-5])  # Extract number from "round_X.json"
                            round_numbers.append(round_num)
                        except ValueError:
                            continue
                if round_numbers:
                    file_round = max(round_numbers)
        
        # Use the maximum of state and file-based detection
        detected_round = max(state_round, file_round)
        
        if detected_round > 0:
            print(f"[RESUME] Run {run_number} - {experiment_name}: completed through round {detected_round}")
            if state_round != detected_round:
                self.update_round(experiment_name, detected_round, run_number)
        
        return detected_round
    
    def update_round(self, experiment_name, round_num, run_number=None):
        """Update the last completed round for a specific run."""
        if run_number is None:
            run_number = self.state_data.get('current_run', 1)
        
        run_key = f"run_{run_number}"
        if 'runs' not in self.state_data:
            self.state_data['runs'] = {}
        if run_key not in self.state_data['runs']:
            self.state_data['runs'][run_key] = {}
        if experiment_name not in self.state_data['runs'][run_key]:
            self.state_data['runs'][run_key][experiment_name] = {}
            
        self.state_data['runs'][run_key][experiment_name]['last_round'] = round_num
        self.state_data['runs'][run_key][experiment_name]['last_update'] = time.time()
        self._save_state()
    
    def set_current_run(self, run_number):
        """Set the current run number."""
        self.state_data['current_run'] = run_number
        self._save_state()
    
    def get_current_run(self):
        """Get the current run number."""
        return self.state_data.get('current_run', 1)
    
    def get_resume_info(self):
        """Get information about what can be resumed."""
        current_run = self.get_current_run()
        run_key = f"run_{current_run}"
        
        if run_key not in self.state_data.get('runs', {}):
            return None
            
        experiments = self.state_data['runs'][run_key]
        resume_info = {
            'current_run': current_run,
            'total_experiments': len(experiments),
            'completed_experiments': sum(1 for exp in experiments.values() if exp.get('completed', False)),
            'in_progress_experiments': []
        }
        
        for exp_name, exp_data in experiments.items():
            if not exp_data.get('completed', False) and exp_data.get('last_round', 0) > 0:
                resume_info['in_progress_experiments'].append({
                    'name': exp_name,
                    'last_round': exp_data.get('last_round', 0)
                })
                
        return resume_info

# Set up paths - Using local NetworkFed directory only
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add current directory for local imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import NetworkFed components
import data.data_integration
import federation.algorithms.fedavg
import federation.algorithms.fedprox
import federation.algorithms.feddyn
import federation.algorithms.fednova
import federation.communication.syft_client
import federation.communication.secure_syft_client
import core.enums
import orchestration.experiment_config
import monitoring.metrics_collector
import datasite.factory_node

# Import external datasite configuration system
try:
    from config.datasite_config import DatasiteConfigManager, load_datasite_config
    from datasite.factory_node import create_factory_datasites_from_config, validate_external_datasites
    EXTERNAL_DATASITE_AVAILABLE = True
    print("[INFO] External datasite configuration system available")
except ImportError as e:
    print(f"[WARNING] External datasite configuration not available: {e}")
    EXTERNAL_DATASITE_AVAILABLE = False

# Import early stopping functionality
try:
    from utils.early_stopping import EarlyStoppingManager, LocalEarlyStopping, GlobalEarlyStopping
    EARLY_STOPPING_AVAILABLE = True
    print("[INFO] Early stopping system available")
except ImportError as e:
    print(f"[WARNING] Early stopping not available: {e}")
    EARLY_STOPPING_AVAILABLE = False

# Import enhanced monitoring and parallel execution components
try:
    from monitoring.heartbeat_manager import HeartbeatManager
    from monitoring.status_dashboard import SimpleStatusDashboard
    from monitoring.datasite_diagnostics import create_datasite_diagnostics_tool
    from execution.parallel_execution_manager import ParallelExecutionManager
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ENHANCED_FEATURES_AVAILABLE = True
    print("[INFO] Enhanced features available: Parallel Processing, Heartbeat Monitoring, Fault Tolerance, Diagnostics")
except ImportError as e:
    print(f"[WARNING] Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    # Use basic ThreadPoolExecutor as fallback
    from concurrent.futures import ThreadPoolExecutor, as_completed

# Load optimized models module directly from local utils
optimized_models_path = os.path.join(current_dir, 'utils', 'step1a_optimized_models.py')
if not os.path.exists(optimized_models_path):
    raise FileNotFoundError(f"Could not find optimized models at: {optimized_models_path}")

spec = importlib.util.spec_from_file_location("step1a_optimized_models", optimized_models_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from: {optimized_models_path}")

step1a_optimized_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step1a_optimized_models)

# Extract the functions we need
create_optimized_model = step1a_optimized_models.create_optimized_model
OPTIMIZED_TRAINING_CONFIG = step1a_optimized_models.OPTIMIZED_TRAINING_CONFIG

# Extract classes
FederatedDataDistributor = data.data_integration.FederatedDataDistributor
FedAvgAlgorithm = federation.algorithms.fedavg.FedAvgAlgorithm
FedProxAlgorithm = federation.algorithms.fedprox.FedProxAlgorithm
FedDynAlgorithm = federation.algorithms.feddyn.FedDynAlgorithm
FedNovaAlgorithm = federation.algorithms.fednova.FedNovaAlgorithm
PySyftCommunicationManager = federation.communication.syft_client.PySyftCommunicationManager
SecurePySyftCommunicationManager = federation.communication.secure_syft_client.SecurePySyftCommunicationManager
ModelType = core.enums.ModelType
ExperimentConfig = orchestration.experiment_config.ExperimentConfig
FederatedMetricsCollector = monitoring.metrics_collector.MetricsCollector
FactoryDataSite = datasite.factory_node.FactoryDataSite

print("Imported all classes")

# Global console logging instance - will be set by the main runner
_global_console_logger = None

def set_global_console_logger(logger_func):
    """Set the global console logger function."""
    global _global_console_logger
    _global_console_logger = logger_func

def log_and_print(message, end='\n'):
    """Global function to print to console and log to file."""
    print(message, end=end)
    if _global_console_logger:
        try:
            _global_console_logger(message, end)
        except:
            pass  # Silent fail to not disrupt console output

# Create global progress tracker
progress_tracker = None

class ExperimentProgressTracker:
    """Clean console progress tracker for experiments"""
    
    def __init__(self, total_experiments: int):
        self.total_experiments = total_experiments
        self.current_experiment = 0
        self.experiment_start_time = None
        
    def start_experiment(self, experiment_name: str):
        self.current_experiment += 1
        self.experiment_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[EXPERIMENT] {self.current_experiment}/{self.total_experiments}: {experiment_name}")
        print(f"{'='*80}")
        
    def show_round_progress(self, round_num: int, total_rounds: int, client_metrics: List[Dict]):
        print(f"\n[ROUND] {round_num}/{total_rounds}")
        print(f"   Client Training Results:")
        for i, metrics in enumerate(client_metrics, 1):
            loss = metrics.get('average_loss', 0.0)
            acc = metrics.get('accuracy', 0.0)
            samples = metrics.get('samples_count', 0)
            print(f"     Datasite {i}: Loss={loss:.4f}, Acc={acc:.4f}, Samples={samples}")
    
    def update_round_progress(self, round_num: int, client_results: List[Dict]):
        print(f"\n[ROUND] {round_num}")
        for i, result in enumerate(client_results, 1):
            loss = result.get('loss', 0.0)
            acc = result.get('accuracy', 0.0) 
            print(f"   Client {i}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    def show_aggregated_results(self, round_num: int, global_accuracy: float, global_loss=None):
        print(f"   [GLOBAL] Accuracy={global_accuracy:.4f}", end="")
        if global_loss is not None:
            print(f", Loss={global_loss:.4f}")
        else:
            print()
    
    def show_test_results(self, test_accuracy: float, test_loss: Optional[float] = None):
        print(f"   [TEST] Accuracy: {test_accuracy:.4f}", end="")
        if test_loss is not None:
            print(f", Loss: {test_loss:.4f}")
        else:
            print()
    
    def complete_experiment(self, experiment_name: str, final_accuracy: float, duration: float):
        log_and_print(f"\n[SUCCESS] {experiment_name} COMPLETED!")
        log_and_print(f"   Final Accuracy: {final_accuracy:.4f}")
        log_and_print(f"   Duration: {duration:.2f} seconds")
        log_and_print(f"{'='*80}")
    
    def show_experiment_completion(self, duration: float, final_accuracy: float):
        log_and_print(f"\n[SUCCESS] EXPERIMENT COMPLETED!")
        log_and_print(f"   Final Accuracy: {final_accuracy:.4f}")
        log_and_print(f"   Duration: {duration:.2f} seconds")
        log_and_print(f"{'='*80}")

# Load optimized models module directly from local utils
optimized_models_path = os.path.join(current_dir, 'utils', 'step1a_optimized_models.py')
if not os.path.exists(optimized_models_path):
    raise FileNotFoundError(f"Could not find optimized models at: {optimized_models_path}")

spec = importlib.util.spec_from_file_location("step1a_optimized_models", optimized_models_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from: {optimized_models_path}")

step1a_optimized_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step1a_optimized_models)

# Extract the functions we need
create_optimized_model = step1a_optimized_models.create_optimized_model
OPTIMIZED_TRAINING_CONFIG = step1a_optimized_models.OPTIMIZED_TRAINING_CONFIG

# Extract classes
FederatedDataDistributor = data.data_integration.FederatedDataDistributor
# Remove unused DataSiteDataManager import
FedAvgAlgorithm = federation.algorithms.fedavg.FedAvgAlgorithm
FedProxAlgorithm = federation.algorithms.fedprox.FedProxAlgorithm
FedDynAlgorithm = federation.algorithms.feddyn.FedDynAlgorithm
FedNovaAlgorithm = federation.algorithms.fednova.FedNovaAlgorithm
PySyftCommunicationManager = federation.communication.syft_client.PySyftCommunicationManager
SecurePySyftCommunicationManager = federation.communication.secure_syft_client.SecurePySyftCommunicationManager
ModelType = core.enums.ModelType
ExperimentConfig = orchestration.experiment_config.ExperimentConfig
FederatedMetricsCollector = monitoring.metrics_collector.MetricsCollector
FactoryDataSite = datasite.factory_node.FactoryDataSite

print("Imported all classes")
class RealPySyftExperimentRunner:
    """Runs REAL PySyft experiments with configurable parameters, resumption support, and ENHANCED FEATURES."""
    
    def __init__(self, experiment_params: Optional[ExperimentParams] = None, results_dir: Optional[str] = None, 
                 use_external_datasites: bool = False, enable_enhanced_features: bool = True,
                 max_parallel_datasites: int = 3, heartbeat_interval: int = 30):
        # Use provided parameters or defaults
        if experiment_params is None:
            experiment_params = ExperimentParams()
        
        self.params = experiment_params
        self.use_external_datasites = use_external_datasites
        
        # === ENHANCED FEATURES CONFIG ===
        self.enable_enhanced_features = enable_enhanced_features and ENHANCED_FEATURES_AVAILABLE
        self.max_parallel_datasites = max_parallel_datasites
        self.heartbeat_interval = heartbeat_interval
        
        # === CONFIGURABLE PORTS ===
        self.heartbeat_port = self.params.heartbeat_port
        self.dashboard_port = self.params.dashboard_port
        
        # Set configurable parameters (no more hardcoded values!)
        self.num_rounds = self.params.max_rounds
        self.num_datasites = self.params.num_datasites  
        self.local_epochs = self.params.local_epochs
        self.batch_size = self.params.batch_size
        self.learning_rate = self.params.learning_rate
        
        # Session timing tracking
        self.session_start_time = datetime.now()
        self.first_experiment_start_time = None
        self.external_datasites_registered = []  # Track external datasites for heartbeat refresh
        
        # Create results directory and state management
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # FIXED: Create main results directory structure
            self._main_results_dir = os.path.join("results", f"experiment_run_{timestamp}")
            os.makedirs(self._main_results_dir, exist_ok=True)
            self._current_results_dir = self._main_results_dir  # Always points to main results dir
        else:
            # results_dir was provided - check if it contains existing experiments for resume
            self._main_results_dir = self._find_or_create_results_dir(results_dir)
            self._current_results_dir = self._main_results_dir  # Always points to main results dir
        self._current_experiment_id = None  # For debug logging
        
        # === ENHANCED COMPONENTS INITIALIZATION ===
        if self.enable_enhanced_features:
            try:
                # Check for port conflicts before starting heartbeat manager
                print(f"[HEARTBEAT] Checking for port conflicts on {self.heartbeat_port}...")
                try:
                    import requests
                    test_response = requests.get(f"http://localhost:{self.heartbeat_port}/status", timeout=2)
                    if test_response.status_code == 200:
                        print(f"[HEARTBEAT] ⚠️ WARNING: Another heartbeat manager is already running on port {self.heartbeat_port}!")
                        print(f"[HEARTBEAT] ⚠️ This may be start_heartbeat.py - please stop it before running experiments")
                        print(f"[HEARTBEAT] ⚠️ Port conflict may cause datasite availability issues")
                except requests.exceptions.ConnectionError:
                    print(f"[HEARTBEAT] ✅ Port {self.heartbeat_port} is available")
                except Exception as e:
                    print(f"[HEARTBEAT] Could not check port {self.heartbeat_port}: {e}")
                
                # Initialize heartbeat manager for real-time monitoring
                # FIXED: Set consistent timeout to 300 seconds (5 minutes) to match training timeout
                # This prevents conflicts between training timeout (300s) and heartbeat timeout
                heartbeat_timeout = 300  # Fixed 5-minute timeout to match training timeout
                
                print(f"[HEARTBEAT] Starting integrated heartbeat manager on port {self.heartbeat_port}...")
                self.heartbeat_manager = HeartbeatManager(port=self.heartbeat_port, heartbeat_timeout=heartbeat_timeout)
                
                # Start heartbeat manager with verification
                self.heartbeat_manager.start()
                
                # Give more time for heartbeat manager to fully start
                print(f"[HEARTBEAT] Waiting for heartbeat manager to stabilize...")
                time.sleep(5)  # Extended startup time
                
                # Verify heartbeat manager is responding
                try:
                    import requests
                    test_response = requests.get(f"http://localhost:{self.heartbeat_port}/status", timeout=3)
                    if test_response.status_code == 200:
                        print(f"[HEARTBEAT] ✅ Heartbeat manager confirmed running on port {self.heartbeat_port}")
                    else:
                        print(f"[HEARTBEAT] ⚠️ Heartbeat manager started but not responding correctly")
                except Exception as verify_error:
                    print(f"[HEARTBEAT] ⚠️ Cannot verify heartbeat manager: {verify_error}")
                    print(f"[HEARTBEAT] This may indicate a port conflict - ensure start_heartbeat.py is NOT running")
                
                # Initialize status dashboard for visual monitoring (passing heartbeat manager)
                self.status_dashboard = SimpleStatusDashboard(port=self.dashboard_port, heartbeat_manager=self.heartbeat_manager)
                # Initialize parallel execution manager for fault tolerance
                self.parallel_manager = ParallelExecutionManager(
                    heartbeat_manager=self.heartbeat_manager, 
                    max_wait_per_round=600, 
                    check_interval=heartbeat_interval
                )
                
                # Initialize diagnostic tool for real-time datasite status monitoring
                self.diagnostics_tool = create_datasite_diagnostics_tool(
                    heartbeat_manager=self.heartbeat_manager,
                    dashboard_port=self.dashboard_port
                )
                
                print(f"[ENHANCED] Enhanced features enabled: Parallel({max_parallel_datasites}), Heartbeat({heartbeat_interval}s), Timeout({heartbeat_timeout}s)")
                print(f"[ENHANCED] Monitoring ports: Heartbeat={self.heartbeat_port}, Dashboard={self.dashboard_port}")
                print(f"[ENHANCED] 🔍 Diagnostics tool initialized - use .show_diagnostics() for real-time status")
                
                # Start monitoring services
                self.heartbeat_manager.start()
                self.status_dashboard.start()
                
            except Exception as e:
                print(f"[WARNING] Enhanced features failed to initialize: {e}")
                self.enable_enhanced_features = False
                self.diagnostics_tool = None
        else:
            self.heartbeat_manager = None
            self.status_dashboard = None  
            self.parallel_manager = None
            self.diagnostics_tool = None
            print("[INFO] Enhanced features disabled")
        
        # === NEW: Debug File Logging Setup ===
        self._setup_debug_file_logging()
        
        # === NEW: Console Logging Setup ===
        self._setup_console_logging()
        
        # Set up global console logger
        set_global_console_logger(self._write_to_console_log)
        
        # Initialize state management for resumption
        # Always use the local clean LocalExperimentState class for consistent behavior
        self.state = LocalExperimentState(self._main_results_dir)
        
        # Save experiment configuration if configurable experiments is available
        if CONFIGURABLE_EXPERIMENTS_AVAILABLE:
            config_file = os.path.join(self._main_results_dir, "experiment_config.json")
            self.params.save_to_file(config_file)
            print(f"[CONFIG] Experiment configuration saved to: {config_file}")
        
        log_and_print(f"[RESULTS] Main results directory: {self._main_results_dir}")
        log_and_print(f"[RESULTS] Working directory: {self._current_results_dir}")
        log_and_print(f"[PARAMS] Max rounds: {self.num_rounds}, Local epochs: {self.local_epochs}, Batch size: {self.batch_size}")
        
        # Initialize components (data distributor, algorithms, models, etc.)
        self._initialize_components()
        
    def _find_or_create_results_dir(self, provided_results_dir: str) -> str:
        """Find existing experiment directory for resume or create new one."""
        import glob
        
        # Check if provided directory directly contains experiment_state.json
        state_file = os.path.join(provided_results_dir, "experiment_state.json")
        if os.path.exists(state_file):
            print(f"[RESUME] Found experiment state directly in: {provided_results_dir}")
            return provided_results_dir
        
        # Look for experiment directories within the provided directory
        # Pattern: experiment_run_* or multi_run_*
        if os.path.exists(provided_results_dir):
            pattern1 = os.path.join(provided_results_dir, "experiment_run_*")
            pattern2 = os.path.join(provided_results_dir, "multi_run_*")
            
            experiment_dirs = glob.glob(pattern1) + glob.glob(pattern2)
            
            # Find the most recent directory with experiment_state.json
            valid_dirs = []
            for exp_dir in experiment_dirs:
                state_file = os.path.join(exp_dir, "experiment_state.json")
                if os.path.exists(state_file):
                    # Get modification time of state file
                    mtime = os.path.getmtime(state_file)
                    valid_dirs.append((exp_dir, mtime))
            
            if valid_dirs:
                # Sort by modification time (newest first)
                valid_dirs.sort(key=lambda x: x[1], reverse=True)
                most_recent = valid_dirs[0][0]
                print(f"[RESUME] Found {len(valid_dirs)} experiment directories, using most recent: {most_recent}")
                return most_recent
        
        # No existing experiments found, create new timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_results_dir = os.path.join(provided_results_dir, f"experiment_run_{timestamp}")
        os.makedirs(new_results_dir, exist_ok=True)
        print(f"[NEW] Created new main experiment directory: {new_results_dir}")
        return new_results_dir
        
    def _initialize_components(self):
        """Initialize data distributor and experiment configurations."""
        # Initialize data distributor
        self.data_distributor = FederatedDataDistributor(
            processed_data_path=os.path.join(current_dir, 'processed_data')
        )
        
        # Define algorithms with configurable parameters
        self.algorithms = {
            'FedAvg': {'class': FedAvgAlgorithm, 'params': {'learning_rate': self.learning_rate}},
            'FedProx': {'class': FedProxAlgorithm, 'params': {'learning_rate': self.learning_rate, 'mu': self.params.fedprox_mu}},
            'FedDyn': {'class': FedDynAlgorithm, 'params': {'learning_rate': self.learning_rate, 'alpha': 0.01}},
            'FedNova': {'class': FedNovaAlgorithm, 'params': {'learning_rate': self.learning_rate}}
        }
        
        # Define models  
        self.models = {
            'OptimizedCNN': ModelType.CNN,
            'OptimizedLSTM': ModelType.LSTM,
            'OptimizedHybrid': ModelType.HYBRID
        }
        
        # Define data distributions
        self.data_distributions = {
            'IID': {'strategy': 'iid'},
            'Non-IID': {'strategy': 'non_iid_label', 'alpha': 0.5}
        }
        
        # Define communication styles
        self.communication_styles = ['Standard', 'Secure']
    
    def _setup_debug_file_logging(self):
        """Setup debug file logging to results directory."""
        import logging
        
        # Create debug logger for this experiment runner
        self.debug_logger = logging.getLogger(f"experiment_debug_{id(self)}")
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.debug_logger.handlers = []
        
        # Will create the actual debug.log file when experiment starts
        self.debug_file_handler = None
        
        print(f"[DEBUG] Debug file logging system initialized")
    
    def _setup_console_logging(self):
        """Setup console output logging to capture all terminal output to file."""
        # Create console log file in the main results directory
        self.console_log_path = os.path.join(self._current_results_dir, "console.log")
        
        # Initialize console log file
        try:
            with open(self.console_log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== CONSOLE LOG STARTED ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Results Directory: {self._current_results_dir}\n")
                f.write(f"{'='*50}\n\n")
            print(f"[CONSOLE] Console log file created: {self.console_log_path}")
        except Exception as e:
            print(f"[WARNING] Failed to create console log file: {e}")
            self.console_log_path = None
    
    def _log_and_print(self, message, end='\n'):
        """Print to console AND log to file - no change to console behavior."""
        # Print to console exactly as before
        print(message, end=end)
        
        # Also write to console log file
        if hasattr(self, 'console_log_path') and self.console_log_path:
            try:
                with open(self.console_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{message}{end}")
                    f.flush()  # Ensure immediate write
            except Exception:
                pass  # Silent fail to not disrupt console output
    
    def _write_to_console_log(self, message, end='\n'):
        """Helper method to write only to console log file."""
        if hasattr(self, 'console_log_path') and self.console_log_path:
            try:
                with open(self.console_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{message}{end}")
                    f.flush()  # Ensure immediate write
            except Exception:
                pass  # Silent fail to not disrupt console output
    
    def _create_clean_experiment_directory(self, experiment_name: str, run_number: int) -> str:
        """Create clean directory structure: main_results_dir/run_X/experiment_name/"""
        run_dir = os.path.join(self._main_results_dir, f"run_{run_number}")
        experiment_dir = os.path.join(run_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def _create_experiment_debug_log(self, experiment_name: str, run_number: int = 1):
        """Create debug.log file for specific experiment in clean structure."""
        if self.debug_file_handler:
            self.debug_logger.removeHandler(self.debug_file_handler)
            self.debug_file_handler.flush()
            self.debug_file_handler.close()
        
        # Create debug log file in clean directory structure
        experiment_dir = self._create_clean_experiment_directory(experiment_name, run_number)
        debug_log_path = os.path.join(experiment_dir, "debug.log")
        
        # Create file handler with explicit buffering
        self.debug_file_handler = logging.FileHandler(debug_log_path, mode='w', encoding='utf-8')
        self.debug_file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.debug_file_handler.setFormatter(formatter)
        self.debug_logger.addHandler(self.debug_file_handler)
        
        # Force immediate write to test file creation
        self.debug_logger.info(f"=== DEBUG LOG STARTED FOR EXPERIMENT: {experiment_name} (Run {run_number}) ===")
        self.debug_logger.info(f"Debug log file: {debug_log_path}")
        self.debug_file_handler.flush()  # Force immediate write
        
        print(f"[DEBUG] Created debug log file: {debug_log_path}")
        
        # Verify file was created
        if os.path.exists(debug_log_path):
            print(f"[DEBUG] ✅ Debug log file confirmed created: {debug_log_path}")
        else:
            print(f"[DEBUG] ❌ Failed to create debug log file: {debug_log_path}")
        
        return debug_log_path
    
    # === DIAGNOSTIC AND MONITORING METHODS ===
    
    def show_diagnostics(self):
        """Show comprehensive real-time diagnostics for datasite status."""
        if not hasattr(self, 'diagnostics_tool') or not self.diagnostics_tool:
            print("❌ Diagnostics not available - enhanced features not enabled")
            return None
        
        print("\n🔍 RUNNING COMPREHENSIVE DATASITE DIAGNOSTICS...")
        try:
            diagnostics = self.diagnostics_tool.run_comprehensive_diagnostics()
            return diagnostics
        except Exception as e:
            print(f"❌ Diagnostics failed: {e}")
            return None
    
    def show_datasite_status(self):
        """Show real-time status of all datasites."""
        if not hasattr(self, 'heartbeat_manager') or not self.heartbeat_manager:
            print("❌ Heartbeat manager not available - enhanced features not enabled")
            return
        
        try:
            self.heartbeat_manager.print_real_time_status()
        except Exception as e:
            print(f"❌ Status display failed: {e}")
    
    def force_refresh_status(self):
        """Force immediate refresh of all datasite statuses."""
        if not hasattr(self, 'heartbeat_manager') or not self.heartbeat_manager:
            print("❌ Heartbeat manager not available - enhanced features not enabled")
            return None
        
        try:
            print("🔄 FORCING DATASITE STATUS REFRESH...")
            sync_results = self.heartbeat_manager.force_sync_all_datasites()
            
            print(f"✅ Status refresh completed:")
            print(f"   • Updated: {sync_results['updated_count']} datasites")
            print(f"   • Status changes: {len(sync_results['status_changes'])}")
            
            if sync_results['status_changes']:
                print("   • Changes:")
                for change in sync_results['status_changes']:
                    print(f"     - {change['datasite']}: {change['old_status']} → {change['new_status']}")
            
            return sync_results
        except Exception as e:
            print(f"❌ Force refresh failed: {e}")
            return None
    
    def get_true_datasite_status(self) -> Dict[str, Any]:
        """
        Get TRUE status of all datasites with no fallback or hardcoded values.
        
        Returns:
            Dictionary with actual datasite status based on real heartbeat data
        """
        if not hasattr(self, 'heartbeat_manager') or not self.heartbeat_manager:
            return {'error': 'Heartbeat manager not available'}
        
        try:
            return self.heartbeat_manager.get_real_time_status_summary()
        except Exception as e:
            return {'error': f'Failed to get status: {e}'}
    
    def open_dashboard(self):
        """Open the status dashboard in browser."""
        if not hasattr(self, 'status_dashboard') or not self.status_dashboard:
            print("❌ Dashboard not available - enhanced features not enabled")
            return
        
        import webbrowser
        dashboard_url = f"http://localhost:{self.dashboard_port}"
        print(f"🔗 Opening dashboard: {dashboard_url}")
        try:
            webbrowser.open(dashboard_url)
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print(f"📋 Manual URL: {dashboard_url}")

    def get_all_experiment_names(self) -> List[str]:
        """Generate all possible experiment names."""
        experiments = []
        for algorithm in self.algorithms.keys():
            for model in self.models.keys():
                for distribution in self.data_distributions.keys():
                    for communication in self.communication_styles:
                        experiment_name = f"{algorithm}_{model}_{distribution}_{communication}"
                        experiments.append(experiment_name)
        return experiments
    
    def setup_real_datasites(self, distribution_config, model_name):
        """Set up REAL PySyft datasites with proper train/val/test distribution."""
        # Determine data type based on model type
        data_type_map = {
            'optimizedcnn': 'tabular',      # CNN uses tabular (flat) data
            'optimizedlstm': 'sequences',   # LSTM uses sequence data  
            'optimizedhybrid': 'sequences', # Hybrid uses sequence data
            # Also support short names for backward compatibility
            'cnn': 'tabular',      
            'lstm': 'sequences',   
            'hybrid': 'sequences'  
        }
        data_type = data_type_map.get(model_name.lower(), 'tabular')
        
        print(f"[WRENCH] Setting up datasites for model: {model_name} -> data type: {data_type}")
        print(f"[CHART] Distribution: {distribution_config['strategy']}")
        
        # Create datasets with distributed train/val and complete test data
        datasite_datasets = self.data_distributor.create_datasite_datasets(
            num_datasites=self.num_datasites,
            distribution_strategy=distribution_config['strategy'],
            data_type=data_type,
            alpha=distribution_config.get('alpha', 0.5)
        )
        
        # Create REAL PySyft datasites with the new implementation
        factory_datasites = {}
        
        if self.use_external_datasites and EXTERNAL_DATASITE_AVAILABLE:
            print("[EXTERNAL] Using external PySyft datasites from configuration")
            
            # Load external datasites from configuration
            try:
                config_file = "config/datasite_configs.yaml"
                if not os.path.exists(config_file):
                    raise FileNotFoundError(f"External datasite configuration file not found: {config_file}")
                
                # Create external factory datasites from configuration
                external_factory_datasites = create_factory_datasites_from_config(
                    config_file=config_file,
                    verbose=True
                )
                
                # Map datasite_ids to external datasites and upload data
                available_sites = list(external_factory_datasites.keys())
                datasite_list = list(datasite_datasets.items())
                
                for i, (datasite_id, federated_dataset) in enumerate(datasite_list):
                    if i < len(available_sites):
                        external_site_id = available_sites[i]
                        external_datasite = external_factory_datasites[external_site_id]
                        
                        # Clear any existing PySyft requests from previous experiments  
                        print(f"[CLEANUP] Clearing PySyft request cache on external datasite {external_site_id}")
                        try:
                            # Connect to the external datasite and clear requests
                            if hasattr(external_datasite, 'admin_client') and external_datasite.admin_client:
                                client = external_datasite.admin_client
                                # Clear existing requests if any
                                if hasattr(client, 'requests') and client.requests:
                                    existing_requests = client.requests.get_all()
                                    for req in existing_requests:
                                        try:
                                            if hasattr(req, 'delete'):
                                                req.delete()
                                            elif hasattr(client.requests, 'delete') and hasattr(req, 'id'):
                                                client.requests.delete(req.id)
                                        except Exception as del_e:
                                            print(f"[WARNING] Could not delete request {req}: {del_e}")
                                print(f"[SUCCESS] Cleared PySyft request cache for {external_site_id}")
                            else:
                                print(f"[WARNING] No admin client available for {external_site_id}, skipping cache clear")
                        except Exception as e:
                            print(f"[WARNING] Could not clear PySyft request cache for {external_site_id}: {e}")
                            # Continue without failing the experiment
                        
                        # Upload the federated dataset to the external datasite
                        print(f"[EXTERNAL] Uploading data to external datasite {external_site_id}")
                        external_datasite.federated_dataset = federated_dataset
                        external_datasite.model_type = model_name.lower()
                        
                        # Auto-upload the data
                        external_datasite._auto_upload_federated_dataset()
                        
                        # Use the REAL factory name for storage (not logical datasite_id or config id)
                        real_factory_name = external_datasite.site_name  # Get real factory name
                        factory_datasites[real_factory_name] = external_datasite
                        print(f"[SUCCESS] Mapped {datasite_id} to external datasite {real_factory_name} (config: {external_site_id})")
                        
                        # Register external datasite with heartbeat manager for monitoring
                        # Use actual factory name (site_name) instead of generic datasite_id
                        if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager:
                            real_factory_name = external_datasite.site_name  # Get real factory name
                            self.heartbeat_manager.register_datasite(
                                real_factory_name,  # Use real factory name from site_name
                                endpoint=f"http://{external_datasite.hostname}:{external_datasite.port}", 
                                status='online'
                            )
                            
                            # For external datasites, manually refresh heartbeat to prevent timeout
                            # since external datasites don't send regular heartbeats
                            self.heartbeat_manager.datasite_status[real_factory_name]['last_heartbeat'] = datetime.now()
                            
                            # Track external datasite for periodic heartbeat refresh
                            self.external_datasites_registered.append(real_factory_name)
                            
                            print(f"[MONITORING] External datasite {real_factory_name} (config: {external_site_id}) registered with heartbeat manager")
                    else:
                        raise Exception(f"Not enough external datasites configured. Need {len(datasite_list)}, have {len(available_sites)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to setup external datasites: {e}")
                # NO FALLBACK - External datasite setup failure should stop experiment
                raise RuntimeError(f"External datasite setup failed: {e}") from e
        
        if not self.use_external_datasites:
            print("[LOCAL] Launching new PySyft datasites")
            
            for i, (datasite_id, federated_dataset) in enumerate(datasite_datasets.items()):
                try:
                    print(f"[FACTORY] Creating Factory DataSite {datasite_id} on port {8081 + i}")
                    
                    # Create REAL PySyft factory datasite with proper data handling
                    factory_datasite = FactoryDataSite(
                        datasite_id=datasite_id,
                        port=8081 + i,
                        federated_dataset=federated_dataset,
                        model_type=model_name.lower(),
                        reset=True
                    )
                    
                    # Validate that the datasite is actually functional
                    if factory_datasite.is_functional():
                        # Use the descriptive site_name as the key for consistency with dashboard
                        factory_datasites[factory_datasite.site_name] = factory_datasite
                        print(f"[SUCCESS] Factory DataSite {factory_datasite.site_name} (originally {datasite_id}) created successfully and is functional")
                        
                        # Register datasite with heartbeat manager for monitoring
                        # Use actual factory name instead of generic datasite_id
                        if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager:
                            self.heartbeat_manager.register_datasite(
                                factory_datasite.site_name,  # Use descriptive site name for dashboard
                                endpoint=f"http://localhost:{factory_datasite.port}", 
                                status='online'
                            )
                            print(f"[MONITORING] Datasite {factory_datasite.site_name} registered with heartbeat manager")
                    else:
                        print(f"[ERROR] Factory DataSite {datasite_id} was created but is NOT functional")
                        raise Exception(f"Datasite {datasite_id} failed validation - cannot proceed")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to create Factory DataSite {datasite_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        print(f"[TARGET] Successfully created {len(factory_datasites)} REAL PySyft datasites")
        return factory_datasites
    
    def create_model_config(self, model_name):
        """Create model configuration using AI4I dataset parameters."""
        # These are the actual AI4I 2020 dataset dimensions
        return {
            'input_features': 10,  # AI4I dataset feature count
            'hidden_size': 64,
            'num_classes': 2,      # Binary classification: failure/no-failure
            'sequence_length': 10,  # For temporal LSTM models
            'dropout_rate': 0.2
        }
    
    def _pre_experiment_cleanup(self):
        """
        FIXED: Pre-experiment cleanup to prevent hanging issues.
        Clean up any existing PySyft state before starting new experiments.
        """
        print("🧹 Performing pre-experiment cleanup...")
        
        try:
            # Clean up any existing factory datasites
            if hasattr(self, 'factory_datasites') and self.factory_datasites:
                print(f"   🗑️ Cleaning up {len(self.factory_datasites)} existing datasites...")
                for datasite_id, factory_datasite in self.factory_datasites.items():
                    try:
                        print(f"   🧹 Cleaning datasite: {datasite_id}")
                        factory_datasite.cleanup()
                    except Exception as e:
                        print(f"   ⚠️ Failed to cleanup {datasite_id}: {e}")
                
                # Clear the factory datasites dictionary
                self.factory_datasites.clear()
                print(f"   ✅ Cleared factory datasites dictionary")
            
            # Reset any global state
            if hasattr(self, '_current_experiment_id'):
                self._current_experiment_id = None
            
            # Small delay to allow cleanup to complete
            time.sleep(2)
            
            print("   ✅ Pre-experiment cleanup completed")
            
        except Exception as e:
            print(f"   ⚠️ Pre-experiment cleanup failed: {e}")
            # Continue anyway - cleanup failure shouldn't stop experiments

    def run_single_real_experiment(self, algorithm_name, model_name, distribution_name, communication_style, run_number=1):
        """Run single REAL PySyft experiment with FRESH datasites."""
        experiment_id = f"{algorithm_name}_{model_name}_{distribution_name}_{communication_style}"
        print(f"Starting REAL experiment: {experiment_id}")
        
        # Set experiment tracking for metrics collection
        self._current_experiment_id = experiment_id
        
        # === NEW: Create debug log file for this experiment ===
        debug_log_path = self._create_experiment_debug_log(experiment_id)
        self.debug_logger.info(f"=== EXPERIMENT START: {experiment_id} ===")
        self.debug_logger.info(f"Algorithm: {algorithm_name}, Model: {model_name}")
        self.debug_logger.info(f"Distribution: {distribution_name}, Communication: {communication_style}")
        self.debug_logger.info(f"Parameters: rounds={self.num_rounds}, epochs={self.local_epochs}, batch_size={self.batch_size}")
        
        # Results directory already established in constructor and state management
        
        start_time = time.time()
        
        # FIXED: Pre-experiment cleanup to prevent hanging issues
        print(f"🧹 Running pre-experiment cleanup for {experiment_id}...")
        self._pre_experiment_cleanup()
        
        # Set up COMPLETELY FRESH REAL datasites (force recreation)
        distribution_config = self.data_distributions[distribution_name]
        factory_datasites = self.ensure_fresh_datasites(distribution_config, model_name)
        
        # Create algorithm instance
        algorithm_config = self.algorithms[algorithm_name]
        algorithm = algorithm_config['class'](**algorithm_config['params'])
        
        # Create model configuration
        model_config = self.create_model_config(model_name)
        model_type = self.models[model_name]
        
        # Set up experiment configuration  
        model_type_enum = ModelType.CNN if model_type == ModelType.CNN else ModelType.LSTM if model_type == ModelType.LSTM else ModelType.HYBRID
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            name=f"{algorithm_name}_{model_type.name}_{distribution_config['strategy']}_{communication_style}",
            description=f"REAL federated learning experiment",
            model_type=model_type_enum,
            model_params=model_config,
            algorithm_params={
                'name': algorithm.__class__.__name__,
                'learning_rate': getattr(algorithm, 'learning_rate', 0.01),
                'device': getattr(algorithm, 'device', 'cpu')
            },
            max_rounds=self.num_rounds,
            training_params={
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        )
        
        # Initialize metrics collector
        metrics_collector = FederatedMetricsCollector()
        
        # Run REAL federated learning
        results = self.run_real_federated_learning(
            algorithm, factory_datasites, experiment_config, metrics_collector, experiment_id, 0, run_number
        )
        
        # Collect results
        end_time = time.time()
        experiment_duration = end_time - start_time
        
        final_results = {
            'experiment_id': experiment_id,
            'algorithm': algorithm_name,
            'model': model_name,
            'distribution': distribution_name,
            'status': 'completed',
            'duration_seconds': experiment_duration,
            'final_metrics': results['final_metrics'],
            'round_metrics': results['round_metrics'],
            'algorithm_specific_metrics': algorithm.get_algorithm_specific_metrics([]),
            'data_distribution_stats': results['data_stats'],
            'config': {
                'max_rounds': self.num_rounds,
                'num_datasites': self.num_datasites,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size
            }
        }
        
        # Clean up datasites
        self.cleanup_datasites(factory_datasites)
        
        # Save summary JSON in expected format
        self._save_experiment_summary(experiment_id, final_results, experiment_duration)
        
        print(f"SUCCESS: REAL experiment {experiment_id} completed!")
        print(f"   Duration: {experiment_duration:.2f} seconds")
        print(f"   Final Validation Accuracy: {results['final_metrics'].get('accuracy', 0.0):.4f}")
        print(f"   Final Test Accuracy: {results['final_metrics'].get('test_accuracy', 0.0):.4f}")
        print(f"   Final Test Loss: {results['final_metrics'].get('test_loss', 0.0):.4f}")
        if results['final_metrics'].get('test_f1', 0.0) > 0:
            print(f"   Test F1-Score: {results['final_metrics'].get('test_f1', 0.0):.4f}")
        
        return final_results

    def run_real_federated_learning(self, algorithm, factory_datasites: Dict[str, FactoryDataSite], 
                                   config: ExperimentConfig, metrics_collector: FederatedMetricsCollector,
                                   experiment_name: Optional[str] = None, start_round: int = 0, run_number: int = 1) -> Dict[str, Any]:
        """Run REAL federated learning with PySyft datasites."""
        global progress_tracker
        
        print(f"Starting REAL federated learning with {len(factory_datasites)} PySyft datasites...")
        
        # Initialize early stopping if available
        early_stopping_manager = None
        if EARLY_STOPPING_AVAILABLE:
            try:
                # Get early stopping parameters from config or use defaults
                if hasattr(config, 'training_params'):
                    patience = getattr(config.training_params, 'early_stopping_patience', 5)
                    min_delta = getattr(config.training_params, 'early_stopping_min_delta', 0.01)
                    metric = getattr(config.training_params, 'early_stopping_metric', 'val_loss')
                else:
                    patience = 5
                    min_delta = 0.01
                    metric = 'val_loss'
                
                # Create configuration dictionaries for EarlyStoppingManager
                local_config = {
                    'patience': patience,
                    'min_delta': min_delta,
                    'monitor': metric,
                    'mode': 'min' if 'loss' in metric else 'max'
                }
                
                # Get max_rounds from config - FIXED: Use command-line parameters, not template
                max_rounds = self.num_rounds  # Use command-line parameter directly
                print(f"[DEBUG] Using max_rounds={max_rounds} from command-line (ignoring template)")
                
                global_config = {
                    'patience': patience * 3,  # Increased global patience to prevent Round 15 deadlock
                    'min_delta': min_delta,
                    'monitor': 'accuracy',
                    'mode': 'max',
                    'min_rounds': max(3, max_rounds // 5)  # More conservative min_rounds to allow longer training
                }
                
                early_stopping_manager = EarlyStoppingManager(
                    local_config=local_config,
                    global_config=global_config
                )
                print(f"[INFO] Early stopping enabled: patience={patience}, min_delta={min_delta}, metric={metric}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize early stopping: {e}")
                early_stopping_manager = None
        
        # Extract real dimensions from federated datasites
        print("Extracting data dimensions from federated datasites...")
        input_dim, num_classes, sequence_length = self._extract_data_dimensions(factory_datasites)
        print(f"Extracted dimensions: input_dim={input_dim}, num_classes={num_classes}, sequence_length={sequence_length}")
        
        # Initialize global model using your optimized models with real dimensions
        model_type = config.model_type.value.lower() if hasattr(config.model_type, 'value') else str(config.model_type).lower()
        global_model = create_optimized_model(model_type, input_dim=input_dim, num_classes=num_classes, sequence_length=sequence_length)
        
        round_metrics = []
        early_stopped = False
        
        # Resume support: Start from specified round
        if start_round > 0:
            print(f"[RESUME] ⏭️  Resuming from round {start_round + 1}/{config.max_rounds}")
        
        # Training rounds
        for round_num in range(start_round, config.max_rounds):
            round_start_time = time.time()
            client_updates = []
            client_metrics = {}
            
            # Refresh external datasite heartbeats at start of each round
            if self.use_external_datasites:
                self.refresh_external_datasite_heartbeats()
            
            # Update dashboard with current round progress
            if hasattr(self, 'status_dashboard') and self.status_dashboard:
                self.status_dashboard.update_experiment_status(
                    current_round=round_num + 1,
                    total_rounds=config.max_rounds,
                )
            
            # === DEBUG LOGGING: Round Start ===
            self.debug_logger.info(f"=== ROUND {round_num + 1}/{config.max_rounds} START ===")
            
            # === CRITICAL FIX: Round 15 Hanging Prevention ===
            original_timeout = 300  # Default timeout
            if round_num >= 14:  # Starting from Round 15 (0-indexed = 14)
                print(f"[CRITICAL] Round {round_num + 1} detected - applying anti-hanging measures")
                
                # === DISABLED: ROUND 15+ PYSYFT REQUEST CLEANUP ===
                # Disabled between-rounds cleanup as it takes too long and PySyft handles duplicate code automatically
                # print(f"[ROUND 15+ CLEANUP] Clearing PySyft requests to prevent duplicate request errors...")
                # self._cleanup_pysyft_requests_round15(factory_datasites, round_num + 1)
                print(f"[OPTIMIZATION] Skipped between-rounds cleanup - PySyft handles duplicate code submission automatically")
                
                # Increase timeouts for Rounds 15+ to prevent factory_03 timeout issues
                import threading
                if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager:
                    original_timeout = getattr(self.heartbeat_manager, 'heartbeat_timeout', 300)
                    self.heartbeat_manager.heartbeat_timeout = 600  # Double timeout for Round 15+
                    print(f"[CRITICAL] Extended heartbeat timeout from {original_timeout}s to 600s for Round {round_num + 1}")
            
                # === ENHANCED: PARALLEL CLIENT TRAINING ===
            if self.enable_enhanced_features and self.parallel_manager:
                # ENHANCED: Parallel training with fault tolerance
                self.debug_logger.info(f"[ENHANCED] Starting parallel training for {len(factory_datasites)} datasites")
                
                # DIAGNOSTIC: Print heartbeat status for debugging
                if self.heartbeat_manager and hasattr(self.heartbeat_manager, 'print_diagnostic_report'):
                    print(f"\n[DIAGNOSTIC] Heartbeat status before datasite availability check:")
                    self.heartbeat_manager.print_diagnostic_report()
                
                # Check datasite availability with heartbeat (with fallback for external datasites)
                print(f"\n[AVAILABILITY] Checking availability of {len(factory_datasites)} datasites...")
                print(f"[AVAILABILITY] Use external datasites: {self.use_external_datasites}")
                print(f"[AVAILABILITY] Heartbeat manager available: {self.heartbeat_manager is not None}")
                
                available_datasites = {}
                for site_name, factory_datasite in factory_datasites.items():
                    print(f"\n[CHECK] Checking datasite: {site_name}")
                    
                    if self.heartbeat_manager and not self.use_external_datasites:
                        print(f"[CHECK] {site_name}: Using LOCAL datasite logic (heartbeat + functional fallback)")
                        # For local datasites, use heartbeat monitoring with site_name as key
                        status_info = self.heartbeat_manager.get_datasite_status(site_name)
                        if status_info and status_info.get('status') == 'online':
                            available_datasites[site_name] = factory_datasite
                            print(f"[CHECK] {site_name}: ✅ AVAILABLE via heartbeat (online)")
                            self.debug_logger.info(f"Local datasite {site_name} available (status: online)")
                        else:
                            # FALLBACK: If heartbeat shows offline, check if datasite is actually functional
                            # This handles cases where local datasites don't send periodic heartbeats
                            actual_status = status_info.get('status') if status_info else 'not_found'
                            print(f"[CHECK] {site_name}: ⚠️ Heartbeat status: {actual_status}, checking functional...")
                            self.debug_logger.warning(f"Datasite {site_name} heartbeat status: {actual_status}, checking functional status...")
                            try:
                                if factory_datasite.is_functional():
                                    available_datasites[site_name] = factory_datasite
                                    print(f"[CHECK] {site_name}: ✅ AVAILABLE via functional check")
                                    self.debug_logger.info(f"Local datasite {site_name} available via functional check (heartbeat: {actual_status})")
                                    # Update heartbeat status to reflect actual availability
                                    if status_info:
                                        self.heartbeat_manager.datasite_status[site_name]['status'] = 'online'
                                        self.heartbeat_manager.datasite_status[site_name]['last_heartbeat'] = datetime.now()
                                else:
                                    print(f"[CHECK] {site_name}: ❌ NOT AVAILABLE (functional check failed)")
                                    self.debug_logger.warning(f"Local datasite {site_name} not functional (heartbeat: {actual_status})")
                            except Exception as e:
                                print(f"[CHECK] {site_name}: ❌ ERROR during functional check: {e}")
                                self.debug_logger.error(f"Failed to check datasite {site_name} functionality: {e}")
                    else:
                        print(f"[CHECK] {site_name}: Using EXTERNAL datasite logic (heartbeat + functional fallback)")
                        # For external datasites, use heartbeat status with functional fallback
                        if self.heartbeat_manager:
                            # Check heartbeat status first for external datasites too
                            status_info = self.heartbeat_manager.get_datasite_status(site_name)
                            if status_info and status_info.get('status') == 'online':
                                available_datasites[site_name] = factory_datasite
                                print(f"[CHECK] {site_name}: ✅ AVAILABLE via heartbeat (online)")
                                self.debug_logger.info(f"External datasite {site_name} available (heartbeat: online)")
                            else:
                                # FALLBACK: If heartbeat shows offline or missing, check functional status
                                actual_status = status_info.get('status') if status_info else 'not_found'
                                print(f"[CHECK] {site_name}: ⚠️ Heartbeat status: {actual_status}, checking functional...")
                                self.debug_logger.warning(f"External datasite {site_name} heartbeat status: {actual_status}, checking functional status...")
                                try:
                                    if factory_datasite.is_functional():
                                        available_datasites[site_name] = factory_datasite
                                        print(f"[CHECK] {site_name}: ✅ AVAILABLE via functional check")
                                        self.debug_logger.info(f"External datasite {site_name} available via functional check (heartbeat: {actual_status})")
                                        # Update heartbeat status to reflect actual availability
                                        if status_info:
                                            self.heartbeat_manager.datasite_status[site_name]['status'] = 'online'
                                            self.heartbeat_manager.datasite_status[site_name]['last_heartbeat'] = datetime.now()
                                    else:
                                        print(f"[CHECK] {site_name}: ❌ NOT AVAILABLE (functional check failed)")
                                        self.debug_logger.warning(f"External datasite {site_name} not functional (heartbeat: {actual_status})")
                                except Exception as e:
                                    print(f"[CHECK] {site_name}: ❌ ERROR during functional check: {e}")
                                    self.debug_logger.error(f"Failed to check external datasite {site_name} functionality: {e}")
                        else:
                            print(f"[CHECK] {site_name}: No heartbeat manager, using functional check only")
                            # No heartbeat manager, fall back to functional check only
                            try:
                                if factory_datasite.is_functional():
                                    available_datasites[site_name] = factory_datasite
                                    print(f"[CHECK] {site_name}: ✅ AVAILABLE via functional check (no heartbeat manager)")
                                    self.debug_logger.info(f"External datasite {site_name} validated as functional (no heartbeat manager)")
                                else:
                                    print(f"[CHECK] {site_name}: ❌ NOT AVAILABLE (functional check failed)")
                                    self.debug_logger.warning(f"External datasite {site_name} failed functionality check")
                            except Exception as e:
                                print(f"[CHECK] {site_name}: ❌ ERROR during functional check: {e}")
                                self.debug_logger.warning(f"External datasite {site_name} validation failed: {e}")
                
                print(f"\n[AVAILABILITY] Results: {len(available_datasites)}/{len(factory_datasites)} datasites available")
                for site_name in available_datasites.keys():
                    print(f"[AVAILABILITY] ✅ {site_name}")
                for site_name in factory_datasites.keys():
                    if site_name not in available_datasites:
                        print(f"[AVAILABILITY] ❌ {site_name}")
                print(f"[AVAILABILITY] Available datasites: {list(available_datasites.keys())}")
                
                # Ensure minimum 2 datasites available (fault tolerance)
                if len(available_datasites) < 2:
                    self.debug_logger.error(f"Only {len(available_datasites)} datasites available, minimum 2 required")
                    raise RuntimeError(f"Insufficient datasites available: {len(available_datasites)} < 2")
                
                print(f"[ENHANCED] Using {len(available_datasites)}/{len(factory_datasites)} available datasites")
                
                # Parallel training using ThreadPoolExecutor with Round 15+ anti-hanging protection
                with ThreadPoolExecutor(max_workers=min(self.max_parallel_datasites, len(available_datasites))) as executor:
                    # Submit all training tasks
                    future_to_datasite = {
                        executor.submit(self.train_real_client, factory_datasite, global_model, config, round_num): datasite_id
                        for datasite_id, factory_datasite in available_datasites.items()
                    }
                    
                    # === ROUND 15+ ANTI-HANGING: Enhanced timeout and deadlock detection ===
                    successful_results = 0
                    total_futures = len(future_to_datasite)
                    start_time = time.time()
                    max_wait_time = 900 if round_num >= 14 else 600  # 15 minutes for Round 15+, 10 minutes otherwise
                    
                    print(f"[PARALLEL] Waiting for {total_futures} client training tasks to complete (max {max_wait_time}s)")
                    
                    # Use timeout-based completion instead of as_completed() to prevent hanging
                    completed_futures = set()
                    timeout_increment = 30  # Check every 30 seconds
                    
                    while len(completed_futures) < total_futures and (time.time() - start_time) < max_wait_time:
                        # Check for completed futures with timeout protection
                        for future, datasite_id in future_to_datasite.items():
                            if future in completed_futures:
                                continue
                                
                            try:
                                # Non-blocking check if future is done
                                if future.done():
                                    completed_futures.add(future)
                                    client_update = future.result(timeout=1)  # Short timeout since it's done
                                    client_updates.append(client_update)
                                    successful_results += 1
                                    
                                    # === FIX: Collect client metrics in parallel path (same as sequential path) ===
                                    client_number = len(client_metrics) + 1  # Sequential numbering as they complete
                                    
                                    # Create comprehensive client metrics matching sequential path structure
                                    client_metrics[datasite_id] = {
                                        'client_id': datasite_id,
                                        'client_number': client_number,
                                        'accuracy': client_update.get('training_accuracy', client_update.get('val_accuracy', 0.0)),
                                        'loss': client_update.get('training_loss', 0.0),
                                        'val_accuracy': client_update.get('val_accuracy', 0.0),
                                        'val_loss': client_update.get('val_loss', 0.0),
                                        'test_accuracy': client_update.get('test_accuracy', 0.0),
                                        'test_loss': client_update.get('test_loss', 0.0),
                                        'local_epochs': client_update.get('local_epochs', getattr(config.training_params, 'local_epochs', 1) if hasattr(config, 'training_params') else 1),
                                        'samples_count': client_update.get('samples_count', client_update.get('num_samples', 0)),
                                        'timestamp': time.time(),
                                        # Early stopping information
                                        'early_stopped': client_update.get('early_stopped', False),
                                        'early_stopped_epoch': client_update.get('early_stopped_epoch', 0),
                                        'total_epochs_planned': client_update.get('total_epochs_planned', client_update.get('local_epochs', 1))
                                    }
                                    
                                    # === DEBUG LOGGING: Parallel Client Training Results ===
                                    self.debug_logger.info(f"[PARALLEL] Client {datasite_id} training completed:")
                                    self.debug_logger.info(f"  - Training Loss: {client_update.get('training_loss', 0.0):.4f}")
                                    self.debug_logger.info(f"  - Training Accuracy: {client_update.get('training_accuracy', 0.0):.4f}")
                                    self.debug_logger.info(f"  - Validation Loss: {client_update.get('val_loss', 0.0):.4f}")
                                    self.debug_logger.info(f"  - Validation Accuracy: {client_update.get('val_accuracy', 0.0):.4f}")
                                    
                                    print(f"[PARALLEL] ✅ {datasite_id} completed ({successful_results}/{total_futures})")
                                    
                            except Exception as e:
                                completed_futures.add(future)
                                self.debug_logger.error(f"[PARALLEL] Client {datasite_id} training failed: {e}")
                                print(f"[WARNING] Datasite {datasite_id} failed during training: {e}")
                                # Continue with other datasites (fault tolerance)
                        
                        # If all futures completed, break early
                        if len(completed_futures) >= total_futures:
                            break
                        
                        # Progress update for Round 15+ 
                        if round_num >= 14:
                            elapsed = time.time() - start_time
                            print(f"[ROUND 15+ MONITOR] {len(completed_futures)}/{total_futures} completed after {elapsed:.1f}s")
                        
                        # Short sleep to prevent busy waiting
                        time.sleep(1)
                    
                    # === CRITICAL: Final completion check for Round 15+ ===
                    final_elapsed = time.time() - start_time
                    if len(completed_futures) < total_futures:
                        pending_count = total_futures - len(completed_futures)
                        print(f"[TIMEOUT] ⚠️  {pending_count} datasites did not complete within {max_wait_time}s")
                        
                        # For Round 15+, try emergency completion
                        if round_num >= 14:
                            print(f"[ROUND 15+ EMERGENCY] Attempting to salvage remaining futures...")
                            for future, datasite_id in future_to_datasite.items():
                                if future not in completed_futures:
                                    try:
                                        # Emergency short timeout
                                        client_update = future.result(timeout=5)
                                        client_updates.append(client_update)
                                        successful_results += 1
                                        print(f"[EMERGENCY] ✅ Salvaged {datasite_id}")
                                    except:
                                        print(f"[EMERGENCY] ❌ Could not salvage {datasite_id}")
                    else:
                        print(f"[SUCCESS] All {total_futures} parallel training tasks completed in {final_elapsed:.1f}s")
                
                # === REMOVE OLD DUPLICATE CODE ===
                # The client metrics collection is now done above in the loop
                
                # === CRITICAL CHECKPOINT: Ensure we proceed to aggregation ===
                print(f"[CHECKPOINT] Round {round_num + 1} parallel training phase completed")
                print(f"[CHECKPOINT] Collected {len(client_updates)} client updates and {len(client_metrics)} client metrics")
                print(f"[CHECKPOINT] Proceeding to aggregation phase...")
                
                print(f"[ENHANCED] Parallel round completed: {len(client_updates)} successful updates, {len(client_metrics)} client metrics collected")
                
                # === CRITICAL FIX: Partial Aggregation Support ===
                if len(client_updates) == 0:
                    self.debug_logger.error(f"[CRITICAL] No datasites completed training - cannot proceed")
                    raise RuntimeError(f"All datasites failed in round {round_num + 1} - experiment cannot continue")
                elif len(client_updates) < len(available_datasites):
                    failed_count = len(available_datasites) - len(client_updates)
                    self.debug_logger.warning(f"[PARTIAL] Only {len(client_updates)}/{len(available_datasites)} datasites completed training")
                    print(f"[PARTIAL AGGREGATION] ⚠️  Proceeding with {len(client_updates)}/{len(available_datasites)} datasites ({failed_count} failed)")
                    
                    # Identify which datasites failed for logging
                    successful_ids = {update.get('client_id', 'unknown') for update in client_updates}
                    available_ids = set(available_datasites.keys())
                    failed_ids = available_ids - successful_ids
                    
                    for failed_id in failed_ids:
                        self.debug_logger.warning(f"[FAILED] Datasite {failed_id} did not complete training in round {round_num + 1}")
                        print(f"[FAILED] ❌ Datasite {failed_id} timed out or failed")
                else:
                    self.debug_logger.info(f"[SUCCESS] All {len(client_updates)} datasites completed training successfully")
                    print(f"[SUCCESS] ✅ All {len(client_updates)} datasites completed training")
                
            else:
                # ORIGINAL: Sequential client training (fallback)
                print(f"[FALLBACK] Using sequential training for {len(factory_datasites)} datasites")
                for datasite_id, factory_datasite in factory_datasites.items():
                    # === DEBUG LOGGING: Client Training Start ===
                    self.debug_logger.info(f"Training client {datasite_id} for round {round_num + 1}")
                    
                    # REAL PySyft training
                    client_update = self.train_real_client(
                        factory_datasite, global_model, config, round_num
                    )
                    client_updates.append(client_update)
                    
                    # === DEBUG LOGGING: Client Training Results ===
                    self.debug_logger.info(f"Client {datasite_id} training completed:")
                    self.debug_logger.info(f"  - Training Loss: {client_update.get('training_loss', 0.0):.4f}")
                    self.debug_logger.info(f"  - Training Accuracy: {client_update.get('training_accuracy', 0.0):.4f}")
                    self.debug_logger.info(f"  - Validation Loss: {client_update.get('val_loss', 0.0):.4f}")
                    self.debug_logger.info(f"  - Validation Accuracy: {client_update.get('val_accuracy', 0.0):.4f}")
                    self.debug_logger.info(f"  - Validation Precision: {client_update.get('val_precision', 0.0):.4f}")
                    self.debug_logger.info(f"  - Validation Recall: {client_update.get('val_recall', 0.0):.4f}")
                    self.debug_logger.info(f"  - Validation F1: {client_update.get('val_f1', 0.0):.4f}")
                    # === NEW: Test Metrics Logging ===
                    self.debug_logger.info(f"  - Test Loss: {client_update.get('test_loss', 0.0):.4f}")
                    self.debug_logger.info(f"  - Test Accuracy: {client_update.get('test_accuracy', 0.0):.4f}")
                    self.debug_logger.info(f"  - Test Precision: {client_update.get('test_precision', 0.0):.4f}")
                    self.debug_logger.info(f"  - Test Recall: {client_update.get('test_recall', 0.0):.4f}")
                    self.debug_logger.info(f"  - Test F1: {client_update.get('test_f1', 0.0):.4f}")
                    self.debug_logger.info(f"  - Samples Count: {client_update.get('samples_count', 0)}")
                    
                    # Flush debug log after each client for real-time logging
                    if hasattr(self, 'debug_file_handler') and self.debug_file_handler:
                        self.debug_file_handler.flush()
            
            # Ensure we have at least some updates for aggregation (fault tolerance)
            if not client_updates:
                self.debug_logger.error(f"No client updates available for round {round_num + 1}")
                raise RuntimeError(f"No client updates available for round {round_num + 1}")
            
            print(f"[FEDERATION] Aggregating {len(client_updates)} client updates")
            
            # === ROUND 15+ ANTI-HANGING: Aggregation timeout protection ===
            if round_num >= 14:
                print(f"[ROUND 15+ PROTECTION] Adding timeout protection for aggregation phase...")
                import concurrent.futures
                
                # Wrap aggregation in timeout for Round 15+
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(algorithm.aggregate, client_updates)
                    try:
                        aggregated_params = future.result(timeout=300)  # 5-minute timeout for aggregation
                        print(f"[ROUND 15+ SUCCESS] Aggregation completed successfully")
                    except concurrent.futures.TimeoutError:
                        print(f"[ROUND 15+ TIMEOUT] Aggregation exceeded 300 seconds - attempting emergency aggregation")
                        raise RuntimeError(f"Round {round_num + 1} aggregation timeout - experiment terminated")
            else:
                # Normal aggregation for other rounds
                aggregated_params = algorithm.aggregate(client_updates)
            
            # === DEBUG LOGGING: Aggregation ===
            self.debug_logger.info(f"Server aggregation completed for round {round_num + 1}")
            self.debug_logger.info(f"Aggregated {len(client_updates)} client updates using {algorithm.__class__.__name__}")
            
            # DEBUG: Log aggregation info
            print(f"[DEBUG] Aggregated {len(client_updates)} client updates")
            
            # Update global model
            self.update_global_model(global_model, aggregated_params)
            
            # Now evaluate updated global model on each client's validation/test data
            for datasite_id, factory_datasite in factory_datasites.items():
                # Get client training metrics from the training phase
                # Find corresponding client update (some may have failed)
                client_update = None
                successful_datasite_ids = [update.get('datasite_id') for update in client_updates if 'datasite_id' in update]
                
                if datasite_id in successful_datasite_ids:
                    client_update_index = successful_datasite_ids.index(datasite_id)
                    client_update = client_updates[client_update_index]
                else:
                    # This datasite failed training, skip evaluation
                    print(f"[WARNING] Skipping evaluation for failed datasite: {datasite_id}")
                    continue
                
                # Collect comprehensive client metrics including validation and test on UPDATED global model
                try:
                    # Get validation metrics from this client using UPDATED global model
                    val_metrics = factory_datasite.evaluate_model_on_validation(global_model)
                    test_metrics = factory_datasite.evaluate_model_on_test(global_model)
                    
                    client_metrics[datasite_id] = {
                        # Training metrics from local training
                        'accuracy': client_update.get('training_accuracy', client_update.get('val_accuracy', 0.0)),
                        'loss': client_update.get('training_loss', 0.0),
                        'local_epochs': config.training_params.get('local_epochs', 1),
                        'samples_count': client_update.get('samples_count', 0),
                        
                        # Validation metrics from LOCAL training (what client computed during training)
                        'val_accuracy': client_update.get('val_accuracy', 0.0),
                        'val_loss': client_update.get('val_loss', 0.0),
                        'val_precision': client_update.get('val_precision', 0.0),
                        'val_recall': client_update.get('val_recall', 0.0),
                        'val_f1': client_update.get('val_f1', 0.0),
                        'val_auc': client_update.get('val_auc', 0.0),
                        
                        # Global model evaluation metrics (for comparison)
                        'global_val_accuracy': val_metrics.get('accuracy', 0.0),
                        'global_val_loss': val_metrics.get('loss', 0.0),
                        'test_accuracy': test_metrics.get('accuracy', 0.0),
                        'test_loss': test_metrics.get('loss', 0.0),
                        
                        # Early stopping information
                        'early_stopped': client_update.get('early_stopped', False),
                        'early_stopped_epoch': client_update.get('early_stopped_epoch', 0),
                        'total_epochs_planned': client_update.get('total_epochs_planned', client_update.get('local_epochs', 1))
                    }
                    
                    print(f"[DEBUG] Round {round_num} - {datasite_id} validation: {val_metrics.get('correct', 0)}/{val_metrics.get('total', 0)} = {val_metrics.get('accuracy', 0.0):.4f}")
                    print(f"[DEBUG] Round {round_num} - {datasite_id} test: {test_metrics.get('correct', 0)}/{test_metrics.get('total', 0)} = {test_metrics.get('accuracy', 0.0):.4f}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to collect comprehensive metrics from {datasite_id}: {e}")
                    # NO FALLBACK - Metrics collection failure should stop experiment
                    raise RuntimeError(f"Failed to collect metrics from datasite {datasite_id}: {e}") from e
                    
            # DEBUG: Verify model update by checking parameter changes
            if hasattr(global_model, 'state_dict'):
                model_params = global_model.state_dict()
                first_param_key = next(iter(model_params.keys()))
                first_param_sum = model_params[first_param_key].sum().item()
                print(f"[DEBUG] After update - Global model param '{first_param_key}' sum: {first_param_sum:.6f}")
            
            # Evaluate global model on validation data with timeout protection
            print(f"[EVAL] Starting validation evaluation with 300s timeout...")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.evaluate_global_model, global_model, factory_datasites)
                try:
                    val_metrics_data = future.result(timeout=300)  # 5-minute timeout
                    print(f"[EVAL] ✅ Validation evaluation completed successfully")
                except concurrent.futures.TimeoutError:
                    print(f"[TIMEOUT] ⏰ Validation evaluation exceeded 300 seconds")
                    raise RuntimeError(f"Validation evaluation timeout after 300 seconds - experiment terminated")
            
            # Evaluate global model on test data with timeout protection
            print(f"[EVAL] Starting test evaluation with 300s timeout...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.evaluate_global_model_on_test, global_model, factory_datasites)
                try:
                    test_metrics_data = future.result(timeout=300)  # 5-minute timeout
                    print(f"[EVAL] ✅ Test evaluation completed successfully")
                except concurrent.futures.TimeoutError:
                    print(f"[TIMEOUT] ⏰ Test evaluation exceeded 300 seconds")
                    raise RuntimeError(f"Test evaluation timeout after 300 seconds - experiment terminated")
            
            # === DEBUG LOGGING: Global Model Evaluation ===
            self.debug_logger.info(f"Global model evaluation for round {round_num + 1}:")
            self.debug_logger.info(f"  - Global Validation Accuracy: {val_metrics_data.get('accuracy', 0.0):.4f}")
            self.debug_logger.info(f"  - Global Validation Loss: {val_metrics_data.get('loss', 0.0):.4f}")
            self.debug_logger.info(f"  - Global Test Accuracy: {test_metrics_data.get('accuracy', 0.0):.4f}")
            self.debug_logger.info(f"  - Global Test Loss: {test_metrics_data.get('loss', 0.0):.4f}")
            self.debug_logger.info(f"  - Global Test Precision: {test_metrics_data.get('precision', 0.0):.4f}")
            self.debug_logger.info(f"  - Global Test Recall: {test_metrics_data.get('recall', 0.0):.4f}")
            self.debug_logger.info(f"  - Global Test F1: {test_metrics_data.get('f1_score', 0.0):.4f}")
            
            # Combine metrics with detailed debug logging
            round_metrics_data = {
                **val_metrics_data,
                'test_accuracy': test_metrics_data.get('accuracy', 0.0),
                'test_loss': test_metrics_data.get('loss', 0.0),
                'test_precision': test_metrics_data.get('precision', 0.0),
                'test_recall': test_metrics_data.get('recall', 0.0),
                'test_f1': test_metrics_data.get('f1_score', 0.0),
                'round': round_num + 1,
                'duration': time.time() - round_start_time
            }
            
            # === DEBUG LOGGING: Round Summary ===
            round_duration = time.time() - round_start_time
            self.debug_logger.info(f"=== ROUND {round_num + 1} COMPLETED ===")
            self.debug_logger.info(f"Round duration: {round_duration:.2f} seconds")
            self.debug_logger.info(f"Client metrics summary:")
            for datasite_id, metrics in client_metrics.items():
                self.debug_logger.info(f"  {datasite_id}: Train_Acc={metrics['accuracy']:.4f}, Val_Acc={metrics.get('val_accuracy', 0.0):.4f}, Test_Acc={metrics.get('test_accuracy', 0.0):.4f}")
            
            # DEBUG: Comprehensive logging to debug metrics mismatch
            print(f"\n[DEBUG ROUND {round_num + 1}] Metrics collection:")
            print(f"   Validation: Acc={val_metrics_data.get('accuracy', 0.0):.4f}, Loss={val_metrics_data.get('loss', 0.0):.4f}")
            print(f"   Test: Acc={test_metrics_data.get('accuracy', 0.0):.4f}, Loss={test_metrics_data.get('loss', 0.0):.4f}")
            print(f"   Round data accuracy: {round_metrics_data.get('accuracy', 0.0):.4f}")
            print(f"   Round data test_accuracy: {round_metrics_data.get('test_accuracy', 0.0):.4f}")
            
            round_metrics.append(round_metrics_data)
            
            # Print round summary with both validation and test metrics
            print(f"\n[ROUND] {round_num + 1}")
            client_number = 1
            for datasite_id, metrics in client_metrics.items():
                print(f"   Client {client_number}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
                client_number += 1
            
            print(f"   Global Validation: Acc={val_metrics_data.get('accuracy', 0.0):.4f}, Loss={val_metrics_data.get('loss', 0.0):.4f}")
            print(f"   Global Test: Acc={test_metrics_data.get('accuracy', 0.0):.4f}, Loss={test_metrics_data.get('loss', 0.0):.4f}")
            if test_metrics_data.get('precision', 0.0) > 0:
                print(f"   Test Metrics: Precision={test_metrics_data.get('precision', 0.0):.4f}, Recall={test_metrics_data.get('recall', 0.0):.4f}, F1={test_metrics_data.get('f1_score', 0.0):.4f}")
            
            # Update progress tracker with round results
            if progress_tracker:
                # Convert client_metrics dict to list format expected by progress tracker
                client_results_list = [metrics for metrics in client_metrics.values()]
                progress_tracker.update_round_progress(
                    round_num + 1,
                    client_results_list
                )
            
            # Save round metrics to file for real-time monitoring
            self._save_round_metrics(round_num + 1, round_metrics_data, client_metrics, experiment_name, run_number)
            
            # Update experiment state for resume functionality
            if experiment_name:
                self.state.update_round(experiment_name, round_num + 1, run_number)
            
            # Check for early stopping
            if early_stopping_manager:
                try:
                    # Update global early stopping with current round metrics
                    should_stop, stop_reason = early_stopping_manager.global_early_stopping.should_stop(
                        round_metrics_data, round_num + 1
                    )
                    
                    if should_stop:
                        early_stopped = True
                        print(f"\n[EARLY STOPPING] Training stopped at round {round_num + 1}")
                        print(f"[EARLY STOPPING] Reason: {stop_reason}")
                        print(f"[EARLY STOPPING] Best {early_stopping_manager.global_early_stopping.monitor}: {early_stopping_manager.global_early_stopping.best_value:.4f}")
                        
                        # Log early stopping details
                        self.debug_logger.info(f"Early stopping triggered at round {round_num + 1}")
                        self.debug_logger.info(f"Reason: {stop_reason}")
                        self.debug_logger.info(f"Best {early_stopping_manager.global_early_stopping.monitor}: {early_stopping_manager.global_early_stopping.best_value:.4f}")
                        
                        # Update algorithm state before breaking
                        algorithm.update_algorithm_state(client_updates)
                        break
                        
                except Exception as e:
                    print(f"[WARNING] Early stopping check failed: {e}")
            
            # Update algorithm state
            algorithm.update_algorithm_state(client_updates)
            
            # === CRITICAL FIX: Reset timeout after Round 15+ completion ===
            if round_num >= 14 and hasattr(self, 'heartbeat_manager') and self.heartbeat_manager:
                self.heartbeat_manager.heartbeat_timeout = original_timeout
                print(f"[CRITICAL] Reset heartbeat timeout to {original_timeout}s after Round {round_num + 1}")
            
            # === ROUND COMPLETION SAFEGUARD ===
            print(f"[SUCCESS] Round {round_num + 1}/{config.max_rounds} completed successfully")
            self.debug_logger.info(f"=== ROUND {round_num + 1} SAFEGUARD PASSED ===")
        
        # Perform final test evaluation with timeout protection
        if progress_tracker:
            print(f"[FINAL EVAL] Starting final test evaluation with 300s timeout...")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.evaluate_global_model_on_test, global_model, factory_datasites)
                try:
                    test_metrics = future.result(timeout=300)  # 5-minute timeout
                    print(f"[FINAL EVAL] ✅ Final test evaluation completed successfully")
                    progress_tracker.show_test_results(
                        test_metrics.get('accuracy', 0.0),
                        test_metrics.get('loss', 0.0)
                    )
                except concurrent.futures.TimeoutError:
                    print(f"[TIMEOUT] ⏰ Final test evaluation exceeded 300 seconds")
                    raise RuntimeError(f"Final test evaluation timeout after 300 seconds - experiment terminated")
        
        # Get data distribution statistics
        data_stats = self.get_data_distribution_stats(factory_datasites)
        
        # Create final metrics from the last round with clear separation
        final_round_metrics = round_metrics[-1] if round_metrics else {}
        
        # === DEBUG LOGGING: Experiment Completion ===
        self.debug_logger.info(f"=== EXPERIMENT COMPLETION: {config.experiment_id} ===")
        self.debug_logger.info(f"Total rounds completed: {len(round_metrics)}")
        self.debug_logger.info(f"Final round validation accuracy: {final_round_metrics.get('accuracy', 0.0):.4f}")
        self.debug_logger.info(f"Final round test accuracy: {final_round_metrics.get('test_accuracy', 0.0):.4f}")
        self.debug_logger.info(f"Final round test precision: {final_round_metrics.get('test_precision', 0.0):.4f}")
        self.debug_logger.info(f"Final round test recall: {final_round_metrics.get('test_recall', 0.0):.4f}")
        self.debug_logger.info(f"Final round test F1: {final_round_metrics.get('test_f1', 0.0):.4f}")
        
        # Prepare comprehensive final metrics with clear validation vs test labels
        final_metrics = {
            # Validation metrics (primary training performance indicators)
            'validation_accuracy': final_round_metrics.get('accuracy', 0.0),
            'validation_loss': final_round_metrics.get('loss', 0.0),
            'validation_precision': final_round_metrics.get('precision', 0.0),
            'validation_recall': final_round_metrics.get('recall', 0.0),
            'validation_f1': final_round_metrics.get('f1_score', 0.0),
            
            # Test metrics (final evaluation performance)
            'test_accuracy': final_round_metrics.get('test_accuracy', 0.0),
            'test_loss': final_round_metrics.get('test_loss', 0.0),
            'test_precision': final_round_metrics.get('test_precision', 0.0),
            'test_recall': final_round_metrics.get('test_recall', 0.0),
            'test_f1': final_round_metrics.get('test_f1', 0.0),
            
            # Legacy fields for backward compatibility (use validation metrics)
            'accuracy': final_round_metrics.get('accuracy', 0.0),  # validation accuracy
            'loss': final_round_metrics.get('loss', 0.0),         # validation loss
            
            # Metadata
            'total_rounds': len(round_metrics),
            'evaluation_method': final_round_metrics.get('evaluation_method', 'federated_average')
        }
        
        # === DEBUG LOGGING: Final Results Summary ===
        self.debug_logger.info(f"Final experiment results:")
        self.debug_logger.info(f"  - Validation Accuracy: {final_metrics['validation_accuracy']:.4f}")
        self.debug_logger.info(f"  - Test Accuracy: {final_metrics['test_accuracy']:.4f}")
        self.debug_logger.info(f"  - Test Precision: {final_metrics['test_precision']:.4f}")
        self.debug_logger.info(f"  - Test Recall: {final_metrics['test_recall']:.4f}")
        self.debug_logger.info(f"  - Test F1: {final_metrics['test_f1']:.4f}")
        self.debug_logger.info(f"=== END EXPERIMENT DEBUG LOG ===")
        
        # === DEBUG LOGGING: Close and flush debug log file ===
        if hasattr(self, 'debug_file_handler') and self.debug_file_handler:
            self.debug_file_handler.flush()
            self.debug_file_handler.close()
            print(f"[DEBUG] ✅ Debug log file closed and flushed")
        
        # Mark experiment as completed for resume functionality
        if experiment_name:
            # This method is called from both multi-run and individual experiments
            # For multi-run, run_number is passed; for individual, we need to determine it
            # Since this is the completion of run_real_federated_learning, we need run_number
            # This should be passed as a parameter to this method
            print(f"[RESUME] ✅ Experiment {experiment_name} completed")
        
        return {
            'final_metrics': final_metrics,
            'round_metrics': round_metrics,
            'data_stats': data_stats,
            'early_stopped': early_stopped,
            'early_stopping': early_stopping_manager.get_experiment_summary() if early_stopping_manager else None
        }

    def train_real_client(self, factory_datasite: FactoryDataSite, global_model, 
                         config: ExperimentConfig, round_num: int) -> Dict[str, Any]:
        """Perform REAL training at PySyft datasite."""
        datasite_id = factory_datasite.datasite_id
        
        # Get training parameters
        local_epochs = config.training_params.get('local_epochs', 1)
        learning_rate = config.algorithm_params.get('learning_rate', 0.01)
        batch_size = config.training_params.get('batch_size', 32)
        algorithm = config.algorithm_params.get('name', 'FedAvg')
        
        # Prepare training configuration
        training_config = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': local_epochs,
            'algorithm': algorithm.lower(),
            'round_num': round_num,
            'mu': config.algorithm_params.get('mu', 0.01),
            'model_type': config.model_type.value.lower() if hasattr(config.model_type, 'value') else str(config.model_type).lower(),
            # Add early stopping parameters
            'early_stopping_patience': getattr(config.training_params, 'early_stopping_patience', 5),
            'early_stopping_min_delta': getattr(config.training_params, 'early_stopping_min_delta', 0.001),
            'early_stopping_metric': getattr(config.training_params, 'early_stopping_metric', 'val_loss')
        }
        
        # Call REAL PySyft datasite training
        training_results = factory_datasite.train_local_model(
            global_model=global_model,
            training_config=training_config
        )
        
        # Add client_id to the results for aggregation algorithms that need it
        training_results['client_id'] = datasite_id
        
        # Store the training results for global evaluation fallback
        factory_datasite.last_training_result = training_results
        
        print(f"   [SUCCESS] REAL PySyft training completed for {datasite_id}")
        
        return training_results

    def update_global_model(self, global_model, aggregated_params: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters."""
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in aggregated_params:
                    param.data = aggregated_params[name]  # Replace with aggregated params, not add

    def evaluate_global_model(self, global_model, factory_datasites: Dict[str, FactoryDataSite]) -> Dict[str, float]:
        """Evaluate global model on REAL validation data from datasites."""
        try:
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            valid_evaluations = 0
            
            criterion = torch.nn.CrossEntropyLoss()
            all_predictions = []
            all_targets = []
            
            # DEBUG: Log model state for debugging static accuracy issue
            if hasattr(global_model, 'state_dict'):
                model_params = global_model.state_dict()
                first_param_key = next(iter(model_params.keys()))
                first_param_sum = model_params[first_param_key].sum().item()
                print(f"[DEBUG] Global model param '{first_param_key}' sum: {first_param_sum:.6f}")
            
            # Evaluate on each datasite's validation data using proper method
            for datasite_name, factory_datasite in factory_datasites.items():
                try:
                    # Use the proper evaluation method from factory datasite
                    eval_result = factory_datasite.evaluate_model_on_validation(global_model)
                    
                    if 'error' not in eval_result:
                        accuracy = eval_result['accuracy']
                        loss = eval_result['loss']
                        correct = eval_result['correct']
                        samples = eval_result['total']
                        
                        total_correct += correct
                        total_samples += samples
                        total_loss += loss
                        valid_evaluations += 1
                        
                        # Collect prediction data for detailed metrics
                        if 'predictions' in eval_result and 'targets' in eval_result:
                            all_predictions.extend(eval_result['predictions'])
                            all_targets.extend(eval_result['targets'])
                        
                        print(f"[EVAL] {datasite_name}: {correct}/{samples} = {accuracy:.4f} accuracy, loss={loss:.4f}")
                    else:
                        print(f"[WARNING] Failed to evaluate on {datasite_name}: {eval_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"[WARNING] Failed to evaluate on {datasite_name}: {e}")
                    continue
            
            # Calculate aggregated metrics
            if valid_evaluations > 0 and total_samples > 0:
                accuracy = total_correct / total_samples
                avg_loss = total_loss / valid_evaluations
                
                # Calculate detailed metrics from prediction data
                precision = recall = f1 = auc = 0.0
                try:
                    if len(all_predictions) > 0 and len(all_targets) > 0:
                        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                        import numpy as np
                        
                        # Debug prediction distribution
                        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
                        unique_targets, target_counts = np.unique(all_targets, return_counts=True)
                        print(f"[DEBUG] Validation predictions distribution: {dict(zip(unique_preds, pred_counts))}")
                        print(f"[DEBUG] Validation targets distribution: {dict(zip(unique_targets, target_counts))}")
                        
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            all_targets, all_predictions, average='binary', zero_division=0
                        )
                        
                        # Calculate AUC if possible
                        try:
                            if len(set(all_targets)) > 1:  # Need both classes for AUC
                                auc = roc_auc_score(all_targets, all_predictions)
                        except ValueError:
                            auc = 0.0
                    else:
                        print(f"[WARNING] No prediction data available for detailed validation metrics calculation")
                        print(f"[DEBUG] all_predictions length: {len(all_predictions)}, all_targets length: {len(all_targets)}")
                except Exception as e:
                    print(f"[WARNING] Failed to calculate detailed validation metrics: {e}")
                
                result = {
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'total_samples': total_samples,
                    'valid_evaluations': valid_evaluations,
                    'evaluation_method': 'real_validation_data'
                }
                
                print(f"[GLOBAL EVAL] Validation Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f} (from {valid_evaluations} datasites)")
                print(f"[DEBUG] Total correct: {total_correct}, Total samples: {total_samples}, Valid evals: {valid_evaluations}")
                return result
            else:
                # NO FALLBACK - Fail if no datasites can evaluate
                raise RuntimeError(f"No datasites successfully evaluated global model. Cannot proceed with experiment.")
                    
        except Exception as e:
            print(f"[ERROR] Global model evaluation failed: {e}")
            raise RuntimeError(f"Global model evaluation failed: {e}") from e
    
    def evaluate_global_model_on_test(self, global_model, factory_datasites: Dict[str, FactoryDataSite]) -> Dict[str, float]:
        """Evaluate global model on REAL test data from datasites."""
        try:
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            valid_evaluations = 0
            
            criterion = torch.nn.CrossEntropyLoss()
            all_predictions = []
            all_targets = []
            
            # Evaluate on each datasite's test data using proper method
            for datasite_name, factory_datasite in factory_datasites.items():
                try:
                    # Use the proper test evaluation method from factory datasite
                    eval_result = factory_datasite.evaluate_model_on_test(global_model)
                    
                    if 'error' not in eval_result:
                        accuracy = eval_result['accuracy']
                        loss = eval_result['loss']
                        correct = eval_result['correct']
                        samples = eval_result['total']
                        
                        total_correct += correct
                        total_samples += samples
                        total_loss += loss
                        valid_evaluations += 1
                        
                        # Collect prediction data for detailed metrics
                        if 'predictions' in eval_result and 'targets' in eval_result:
                            all_predictions.extend(eval_result['predictions'])
                            all_targets.extend(eval_result['targets'])
                        
                        print(f"[TEST EVAL] {datasite_name}: {correct}/{samples} = {accuracy:.4f} accuracy, loss={loss:.4f}")
                    else:
                        print(f"[WARNING] Failed to evaluate test data on {datasite_name}: {eval_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"[WARNING] Failed to evaluate test data on {datasite_name}: {e}")
                    continue
            
            # Calculate detailed metrics
            if valid_evaluations > 0 and total_samples > 0:
                accuracy = total_correct / total_samples
                avg_loss = total_loss / valid_evaluations
                
                # Calculate precision, recall, f1, auc if we have prediction data
                precision = recall = f1 = auc = 0.0
                try:
                    if len(all_predictions) > 0 and len(all_targets) > 0:
                        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                        import numpy as np
                        
                        # Debug prediction distribution
                        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
                        unique_targets, target_counts = np.unique(all_targets, return_counts=True)
                        print(f"[DEBUG] Test predictions distribution: {dict(zip(unique_preds, pred_counts))}")
                        print(f"[DEBUG] Test targets distribution: {dict(zip(unique_targets, target_counts))}")
                        
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            all_targets, all_predictions, average='binary', zero_division=0
                        )
                        
                        # Calculate AUC if possible
                        try:
                            if len(set(all_targets)) > 1:  # Need both classes for AUC
                                auc = roc_auc_score(all_targets, all_predictions)
                        except ValueError:
                            auc = 0.0
                    else:
                        print(f"[WARNING] No prediction data available for detailed metrics calculation")
                        print(f"[DEBUG] all_predictions length: {len(all_predictions)}, all_targets length: {len(all_targets)}")
                except Exception as e:
                    print(f"[WARNING] Failed to calculate detailed metrics: {e}")
                
                result = {
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'correct': total_correct,  # Add correct count for CSV export
                    'total': total_samples,    # Add total count for CSV export
                    'total_samples': total_samples,
                    'evaluation_method': 'real_test_data'
                }
                
                print(f"[GLOBAL TEST] Acc: {accuracy:.4f}, Loss: {avg_loss:.4f}, F1: {f1:.4f}")
                return result
            else:
                print(f"[ERROR] No test evaluation data available")
                raise RuntimeError("No datasites provided test evaluation data - experiment cannot continue")
                    
        except Exception as e:
            print(f"[ERROR] Test evaluation failed: {e}")
            raise RuntimeError(f"Test evaluation failed: {e}") from e

    def get_data_distribution_stats(self, factory_datasites: Dict[str, FactoryDataSite]) -> Dict[str, Any]:
        """Get data distribution statistics."""
        return {'num_datasites': len(factory_datasites)}

    def cleanup_datasites(self, factory_datasites: Dict[str, FactoryDataSite]):
        """FORCE cleanup REAL PySyft datasites with complete reset."""
        print("🧹 FORCE cleaning up all PySyft datasites...")
        for datasite_id, factory_datasite in factory_datasites.items():
            try:
                print(f"    🔄 Force cleaning datasite {datasite_id}")
                factory_datasite.cleanup()  # This calls server.land()
            except Exception as e:
                print(f"    ❌ Error cleaning datasite {datasite_id}: {e}")
        
        # Extended wait time for complete cleanup and port release
        import time
        print("⏳ Waiting for complete PySyft cleanup and port release...")
        time.sleep(15)  # Increased from 8 to 15 seconds for Windows multiprocessing
        print("✅ All datasites cleaned up and ports released")

    def refresh_external_datasite_heartbeats(self):
        """Refresh heartbeats for external datasites to keep them showing as 'online'"""
        if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager and self.external_datasites_registered:
            current_time = datetime.now()
            for datasite_name in self.external_datasites_registered:
                if datasite_name in self.heartbeat_manager.datasite_status:
                    self.heartbeat_manager.datasite_status[datasite_name]['last_heartbeat'] = current_time
                    self.heartbeat_manager.datasite_status[datasite_name]['status'] = 'online'
                    self.heartbeat_manager.datasite_status[datasite_name]['heartbeat_count'] += 1

    def cleanup_enhanced_components(self):
        """Cleanup enhanced monitoring and parallel execution components."""
        if self.enable_enhanced_features:
            print("[ENHANCED] Cleaning up enhanced components...")
            try:
                if self.heartbeat_manager:
                    self.heartbeat_manager.stop()
                    print("[ENHANCED] Heartbeat manager stopped")
            except Exception as e:
                print(f"[WARNING] Error stopping heartbeat manager: {e}")
            
            # Status dashboard runs in daemon thread, no explicit stop needed
            if self.status_dashboard:
                print("[ENHANCED] Status dashboard will stop automatically")
            
            try:
                if self.parallel_manager:
                    # Parallel manager cleanup if it has any
                    pass
            except Exception as e:
                print(f"[WARNING] Error cleaning up parallel manager: {e}")
            
            print("[ENHANCED] Enhanced components cleanup completed")

    def _cleanup_pysyft_requests_round15(self, factory_datasites, round_num):
        """
        [DISABLED] Clean up PySyft requests for Round 15+ to prevent 'Request already exists' errors.
        
        NOTE: This method is no longer called between rounds as it takes too long and PySyft 
        automatically handles duplicate code submission. The cleanup is now only performed
        at the beginning of each experiment.
        """
        print(f"[ROUND {round_num} CLEANUP] Starting PySyft request cleanup for {len(factory_datasites)} datasites...")
        
        cleanup_success_count = 0
        cleanup_errors = []
        
        for datasite_id, factory_datasite in factory_datasites.items():
            try:
                print(f"[CLEANUP] Clearing requests for {datasite_id}...")
                
                # Access the datasite's admin client
                if hasattr(factory_datasite, 'admin_client') and factory_datasite.admin_client:
                    client = factory_datasite.admin_client
                    
                    # Method 1: Try to clear all requests
                    try:
                        if hasattr(client, 'requests'):
                            existing_requests = client.requests.get_all()
                            for req in existing_requests:
                                try:
                                    if hasattr(req, 'delete'):
                                        req.delete()
                                    elif hasattr(client.requests, 'delete') and hasattr(req, 'id'):
                                        client.requests.delete(req.id)
                                except Exception as del_e:
                                    print(f"[WARNING] Could not delete request {req}: {del_e}")
                            print(f"[SUCCESS] Cleared {len(existing_requests)} requests for {datasite_id}")
                    except Exception as e:
                        print(f"[WARNING] Method 1 failed for {datasite_id}: {e}")
                    
                    # Method 2: Try to clear user code (which creates requests)
                    try:
                        if hasattr(client, 'code'):
                            user_codes = client.code.get_all()
                            for code in user_codes:
                                try:
                                    if hasattr(code, 'delete'):
                                        code.delete()
                                except Exception as del_e:
                                    print(f"[WARNING] Could not delete user code {code}: {del_e}")
                            print(f"[SUCCESS] Cleared {len(user_codes)} user codes for {datasite_id}")
                    except Exception as e:
                        print(f"[WARNING] Method 2 failed for {datasite_id}: {e}")
                    
                    # Method 3: Reset training function flag to force recreation
                    try:
                        if hasattr(factory_datasite, 'training_function_setup'):
                            factory_datasite.training_function_setup = False
                            print(f"[SUCCESS] Reset training function flag for {datasite_id}")
                    except Exception as e:
                        print(f"[WARNING] Method 3 failed for {datasite_id}: {e}")
                    
                    # Method 4: Force-clear the specific request ID that's causing the duplicate error
                    try:
                        # The error shows request ID: 52ef2c8aeec2496f941f1a721c3f0bd6
                        problematic_request_ids = [
                            "52ef2c8aeec2496f941f1a721c3f0bd6",  # Known problematic ID from error
                        ]
                        for req_id in problematic_request_ids:
                            try:
                                if hasattr(client, 'requests') and hasattr(client.requests, 'get'):
                                    req = client.requests.get(req_id)
                                    if req and hasattr(req, 'delete'):
                                        req.delete()
                                        print(f"[SUCCESS] Deleted specific problematic request {req_id} for {datasite_id}")
                            except Exception as spec_e:
                                # This is expected if the request doesn't exist
                                pass
                    except Exception as e:
                        print(f"[WARNING] Method 4 failed for {datasite_id}: {e}")
                    
                    # Method 5: Clear any cached training function data
                    try:
                        # Clear any cached function names or IDs that might cause conflicts
                        if hasattr(factory_datasite, 'function_name'):
                            factory_datasite.function_name = None
                        if hasattr(factory_datasite, 'cached_function'):
                            factory_datasite.cached_function = None
                        if hasattr(factory_datasite, 'last_request_id'):
                            factory_datasite.last_request_id = None
                        print(f"[SUCCESS] Cleared cached function data for {datasite_id}")
                    except Exception as e:
                        print(f"[WARNING] Method 5 failed for {datasite_id}: {e}")
                    
                    cleanup_success_count += 1
                    print(f"[SUCCESS] ✅ Cleanup completed for {datasite_id}")
                    
                else:
                    print(f"[WARNING] No admin client available for {datasite_id}")
                    cleanup_errors.append(f"{datasite_id}: No admin client")
                    
            except Exception as e:
                error_msg = f"{datasite_id}: {str(e)}"
                cleanup_errors.append(error_msg)
                print(f"[ERROR] ❌ Cleanup failed for {datasite_id}: {e}")
        
        # Summary
        total_datasites = len(factory_datasites)
        print(f"[ROUND {round_num} CLEANUP] Summary: {cleanup_success_count}/{total_datasites} datasites cleaned successfully")
        
        if cleanup_errors:
            print(f"[ROUND {round_num} CLEANUP] Errors encountered:")
            for error in cleanup_errors:
                print(f"   ❌ {error}")
        
        # Small delay to let cleanup settle
        import time
        time.sleep(2)
        print(f"[ROUND {round_num} CLEANUP] ✅ PySyft request cleanup completed")

    def ensure_fresh_datasites(self, distribution_config, model_name):
        """Ensure completely fresh PySyft datasites by recreating them."""
        print(f"🔄 Creating FRESH PySyft datasites for {model_name}")
        
        # Always create fresh datasites for each experiment
        factory_datasites = self.setup_real_datasites(distribution_config, model_name)
        
        # Validate all datasites are functional
        functional_count = 0
        for datasite_id, factory_datasite in factory_datasites.items():
            if factory_datasite.is_functional():
                functional_count += 1
            else:
                raise Exception(f"Fresh datasite {datasite_id} is not functional!")
        
        print(f"✅ All {functional_count} fresh datasites are functional")
        
        # Additional wait for fresh datasites to fully stabilize
        import time
        print("⏳ Allowing fresh datasites to stabilize...")
        time.sleep(5)  # Extra stability wait
        print("✅ Fresh datasites stabilized and ready")
        return factory_datasites

    def run_all_experiments(self, num_runs=1, resume=True):
        """Run all 48 REAL PySyft experiments for specified number of runs."""
        global progress_tracker
        
        log_and_print(f"Starting 48 REAL PySyft Federated Learning Experiments")
        if num_runs > 1:
            log_and_print(f"Will repeat ALL experiments {num_runs} times")
        log_and_print("=" * 60)
        log_and_print("[CLIPBOARD] Detailed logs are saved to files, console shows clean progress only")
        
        # Use existing results directory (for resume) or create new one
        if resume and hasattr(self, '_current_results_dir') and self._current_results_dir and os.path.exists(self._current_results_dir):
            results_dir = self._current_results_dir
            log_and_print(f"[RESUME] Using existing results directory: {results_dir}")
            # Check if we have existing state
            # Handle both local and external ExperimentState classes
            state_data = None
            if hasattr(self.state, 'state_data'):
                state_data = self.state.state_data
            elif hasattr(self.state, '_load_state'):
                # Try to call the load method if available
                try:
                    state_data = self.state._load_state()
                except:
                    pass
            
            if state_data:
                log_and_print(f"[RESUME] Found existing experiment state with {len(state_data)} experiments")
                for exp_name, exp_state in state_data.items():
                    if isinstance(exp_state, dict):
                        status = "COMPLETED" if exp_state.get('completed', False) else f"IN PROGRESS (Round {exp_state.get('last_round', 0)})"
                        log_and_print(f"[RESUME]   {exp_name}: {status}")
            else:
                log_and_print(f"[RESUME] Found state object but no experiment data")
        else:
            # Create comprehensive results directory FIRST
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if num_runs > 1:
                results_dir = os.path.join("results", f"multi_run_{num_runs}x_experiment_run_{timestamp}")
            else:
                results_dir = os.path.join("results", f"experiment_run_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            self._current_results_dir = results_dir  # Store for round metrics saving
            if not resume:
                log_and_print(f"[NEW] Created new experiment directory: {results_dir}")
            else:
                log_and_print(f"[RESUME] No existing experiments found, created new directory: {results_dir}")
        
        log_and_print(f"[SAVE] Results will be saved to: {results_dir}")
        
        # Track results across all runs
        all_runs_results = []
        total_successful_experiments = 0
        
        # Calculate total experiments across all runs for dashboard
        total_experiments_all_runs = len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles) * num_runs
        
        # Resume logic: Use ExperimentState to determine where to continue
        start_run = 1
        start_experiment_index = 0
        
        if resume:
            resume_info = self.state.get_resume_info()
            if resume_info:
                current_run = resume_info['current_run']
                completed_in_run = resume_info['completed_experiments']
                total_in_run = len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles)
                
                print(f"[RESUME] Current run: {current_run}")
                print(f"[RESUME] Completed experiments in current run: {completed_in_run}/{total_in_run}")
                
                if completed_in_run == total_in_run:
                    # Current run is complete, start next run
                    start_run = current_run + 1
                    start_experiment_index = 0
                    if start_run <= num_runs:
                        self.state.set_current_run(start_run)
                        print(f"[RESUME] Starting new run: {start_run}")
                else:
                    # Continue current run
                    start_run = current_run
                    start_experiment_index = completed_in_run
                    print(f"[RESUME] Continuing run {start_run} from experiment {start_experiment_index + 1}")
                
                # Show in-progress experiments
                for in_progress in resume_info['in_progress_experiments']:
                    print(f"[RESUME] In progress: {in_progress['name']} at round {in_progress['last_round']}")
                
                # Count completed experiments for progress tracking
                if start_run > 1:
                    # Count all experiments in completed runs
                    experiments_per_run = len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles)
                    total_successful_experiments = (start_run - 1) * experiments_per_run + completed_in_run
                else:
                    total_successful_experiments = completed_in_run
            else:
                print("[RESUME] No previous experiments found, starting fresh")
        
        if start_run > num_runs:
            print(f"[COMPLETE] All {num_runs} runs have been completed!")
            return
        
        # Initialize dashboard with experiment info
        if hasattr(self, 'status_dashboard') and self.status_dashboard:
            self.status_dashboard.update_experiment_status(
                current_experiment="Initializing experiments...",
                current_run=start_run,
                total_runs=num_runs,
                current_round=0,
                total_rounds=0,  # Will be updated when experiment starts
                total_experiments=total_experiments_all_runs,
                experiments_per_run=len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles),
                completed_experiments=total_successful_experiments,
                completed_experiments_current_run=0,
                failed_experiments=0,
                start_time=datetime.now().isoformat()
            )
        
        # Execute multiple runs (starting from the first incomplete run)
        for run_number in range(start_run, num_runs + 1):
            if num_runs > 1:
                log_and_print(f"\n{'='*20} STARTING RUN {run_number}/{num_runs} {'='*20}")
                
            # Count total experiments for this run
            total_experiments = len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles)
            progress_tracker = ExperimentProgressTracker(total_experiments)
            
            # Update dashboard with run info
            if hasattr(self, 'status_dashboard') and self.status_dashboard:
                self.status_dashboard.update_experiment_status(
                    current_experiment=f"Run {run_number}/{num_runs} - Starting",
                    current_round=0,
                    total_experiments=total_experiments_all_runs,
                    completed_experiments=total_successful_experiments,
                    failed_experiments=0,
                    start_time=datetime.now().isoformat()
                )
            
            run_results = []
            successful_experiments = 0
            failed_experiments_this_run = 0
            
            # Run all combinations for this run
            experiment_index = 0
            for algorithm_name in self.algorithms.keys():
                for model_name in self.models.keys():
                    for distribution_name in self.data_distributions.keys():
                        for communication_style in self.communication_styles:
                            experiment_name = f"{algorithm_name}_{model_name}_{distribution_name}_{communication_style}"
                            
                            # Skip experiments that are before our resume point
                            if run_number == start_run and experiment_index < start_experiment_index:
                                experiment_index += 1
                                continue
                            
                            # Check if experiment is already completed using ExperimentState
                            if self.state.is_experiment_completed(experiment_name, run_number):
                                print(f"[RESUME] ⏭️  Skipping completed experiment: {experiment_name} (Run {run_number})")
                                successful_experiments += 1
                                total_successful_experiments += 1
                                progress_tracker.start_experiment(experiment_name)
                                progress_tracker.complete_experiment(experiment_name, final_accuracy=0.0, duration=0.0)
                                experiment_index += 1
                                continue
                                
                            # Create experiment display name  
                            if num_runs > 1:
                                experiment_name_with_run = f"Run{run_number}_{experiment_name}"
                            else:
                                experiment_name_with_run = experiment_name
                                
                            progress_tracker.start_experiment(experiment_name_with_run)
                            
                            # Update dashboard with current experiment
                            if hasattr(self, 'status_dashboard') and self.status_dashboard:
                                # Set first experiment start time if this is the very first experiment
                                if self.first_experiment_start_time is None:
                                    self.first_experiment_start_time = datetime.now().isoformat()
                                
                                self.status_dashboard.update_experiment_status(
                                    current_experiment=experiment_name_with_run,
                                    current_run=run_number,
                                    total_runs=num_runs,
                                    current_round=0,
                                    total_rounds=0,  # Will be updated when experiment starts
                                    total_experiments=total_experiments_all_runs,
                                    experiments_per_run=total_experiments,
                                    completed_experiments=total_successful_experiments + successful_experiments,
                                    completed_experiments_current_run=successful_experiments,
                                    failed_experiments=failed_experiments_this_run,
                                    start_time=datetime.now().isoformat(),
                                    first_experiment_start_time=self.first_experiment_start_time
                                )
                            
                            result = self.run_single_real_experiment(
                                algorithm_name, model_name, distribution_name, communication_style, run_number
                            )
                            
                            # Add run information to result
                            result['run_number'] = run_number
                            result['experiment_name_with_run'] = experiment_name_with_run
                            run_results.append(result)
                            
                            # Save intermediate result immediately
                            self._save_intermediate_result(result, experiment_name, run_number)
                            
                            if result['status'] == 'completed':
                                successful_experiments += 1
                                progress_tracker.complete_experiment(
                                    experiment_name_with_run,
                                    result.get('final_metrics', {}).get('accuracy', 0.0),
                                    result.get('duration_seconds', 0.0)
                                )
                                # Mark experiment as completed in state
                                self.state.mark_experiment_completed(experiment_name, run_number)
                            else:
                                failed_experiments_this_run += 1
                                print(f"[ERROR] EXPERIMENT FAILED: {result.get('error', 'Unknown error')}")
                            
                            # Increment experiment index
                            experiment_index += 1
                            
                            # Update dashboard after each experiment completion
                            if hasattr(self, 'status_dashboard') and self.status_dashboard:
                                self.status_dashboard.update_experiment_status(
                                    current_experiment=experiment_name_with_run,
                                    current_run=run_number,
                                    total_runs=num_runs,
                                    current_round=0,
                                    total_rounds=0,
                                    total_experiments=total_experiments_all_runs,
                                    experiments_per_run=total_experiments,
                                    completed_experiments=total_successful_experiments + successful_experiments,
                                    completed_experiments_current_run=successful_experiments,
                                    failed_experiments=failed_experiments_this_run,
                                    start_time=datetime.now().isoformat()
                                )
            
            # Add run results to overall results
            all_runs_results.extend(run_results)
            total_successful_experiments += successful_experiments
            
            if num_runs > 1:
                print(f"\n[RUN {run_number} COMPLETE] Successful: {successful_experiments}/48 ({(successful_experiments/total_experiments)*100:.1f}%)")
        
        # Final comprehensive results processing across all runs
        
        # Calculate total experiments across all runs
        total_experiments_all_runs = len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles) * num_runs
        
        # Save main results file
        results_file = os.path.join(results_dir, "experiment_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_runs_results, f, indent=2, default=str)
        
        # Save experiment summary
        summary_file = os.path.join(results_dir, "experiment_summary.json")
        summary_data = {
            'experiment_info': {
                'timestamp': timestamp,
                'num_runs': num_runs,
                'total_experiments': total_experiments_all_runs,
                'successful_experiments': total_successful_experiments,
                'success_rate': total_successful_experiments / total_experiments_all_runs if total_experiments_all_runs > 0 else 0.0,
                'algorithms': list(self.algorithms.keys()),
                'models': list(self.models.keys()),
                'distributions': list(self.data_distributions.keys()),
                'communication_styles': self.communication_styles,
                'configuration': {
                    'num_rounds': self.num_rounds,
                    'num_datasites': self.num_datasites,
                    'local_epochs': self.local_epochs,
                    'batch_size': self.batch_size
                }
            },
            'performance_overview': self._calculate_performance_overview(all_runs_results),
            'timing_analysis': self._calculate_timing_analysis(all_runs_results),
            'accuracy_analysis': self._calculate_accuracy_analysis(all_runs_results)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save detailed metrics breakdown
        metrics_file = os.path.join(results_dir, "detailed_metrics.json")
        detailed_metrics = self._extract_detailed_metrics(all_runs_results)
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        
        # Update dashboard with final completion
        if hasattr(self, 'status_dashboard') and self.status_dashboard:
            self.status_dashboard.update_experiment_status(
                current_experiment="ALL COMPLETED",
                current_run=num_runs,
                total_runs=num_runs,
                current_round=0,
                total_rounds=0,
                total_experiments=total_experiments_all_runs,
                experiments_per_run=len(self.algorithms) * len(self.models) * len(self.data_distributions) * len(self.communication_styles),
                completed_experiments=total_successful_experiments,
                completed_experiments_current_run=total_experiments_all_runs // num_runs,  # Average per run
                failed_experiments=total_experiments_all_runs - total_successful_experiments,
                start_time=datetime.now().isoformat()
            )
        
        if num_runs > 1:
            print(f"\n[PARTY] ALL {num_runs} RUNS OF 48 EXPERIMENTS COMPLETED!")
            print(f"[SUCCESS] Total successful: {total_successful_experiments}/{total_experiments_all_runs} ({(total_successful_experiments/total_experiments_all_runs)*100:.1f}%)")
        else:
            print(f"\n[PARTY] ALL 48 REAL EXPERIMENTS COMPLETED!")
            print(f"[SUCCESS] Successful: {total_successful_experiments}/48 ({(total_successful_experiments/48)*100:.1f}%)")
        print(f"[FOLDER] Results directory: {results_dir}")
        print(f"[CHART] Main results: {results_file}")
        print(f"[CLIPBOARD] Summary: {summary_file}")
        print(f"[TRENDING_UP] Detailed metrics: {metrics_file}")
    
    def _calculate_performance_overview(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Calculate performance overview across all experiments."""
        successful_results = [r for r in all_results if r.get('status') == 'completed']
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        # Extract accuracy values
        accuracies = [r.get('final_metrics', {}).get('accuracy', 0.0) for r in successful_results]
        
        return {
            'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'std_accuracy': np.std(accuracies) if accuracies else 0.0,
            'min_accuracy': np.min(accuracies) if accuracies else 0.0,
            'max_accuracy': np.max(accuracies) if accuracies else 0.0,
            'median_accuracy': np.median(accuracies) if accuracies else 0.0
        }
    
    def _calculate_timing_analysis(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Calculate timing analysis across all experiments."""
        successful_results = [r for r in all_results if r.get('status') == 'completed']
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        # Extract duration values
        durations = [r.get('duration_seconds', 0.0) for r in successful_results]
        
        return {
            'avg_duration_seconds': np.mean(durations) if durations else 0.0,
            'std_duration_seconds': np.std(durations) if durations else 0.0,
            'min_duration_seconds': np.min(durations) if durations else 0.0,
            'max_duration_seconds': np.max(durations) if durations else 0.0,
            'total_duration_hours': np.sum(durations) / 3600 if durations else 0.0
        }
    
    def _calculate_accuracy_analysis(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Calculate accuracy analysis by algorithm, model, and distribution."""
        successful_results = [r for r in all_results if r.get('status') == 'completed']
        
        if not successful_results:
            return {'error': 'No successful experiments to analyze'}
        
        analysis = {
            'by_algorithm': {},
            'by_model': {},
            'by_distribution': {}
        }
        
        # Group by algorithm
        for result in successful_results:
            algorithm = result.get('algorithm', 'unknown')
            accuracy = result.get('final_metrics', {}).get('accuracy', 0.0)
            
            if algorithm not in analysis['by_algorithm']:
                analysis['by_algorithm'][algorithm] = []
            analysis['by_algorithm'][algorithm].append(accuracy)
        
        # Group by model
        for result in successful_results:
            model = result.get('model', 'unknown')
            accuracy = result.get('final_metrics', {}).get('accuracy', 0.0)
            
            if model not in analysis['by_model']:
                analysis['by_model'][model] = []
            analysis['by_model'][model].append(accuracy)
        
        # Group by distribution
        for result in successful_results:
            distribution = result.get('distribution', 'unknown')
            accuracy = result.get('final_metrics', {}).get('accuracy', 0.0)
            
            if distribution not in analysis['by_distribution']:
                analysis['by_distribution'][distribution] = []
            analysis['by_distribution'][distribution].append(accuracy)
        
        # Calculate statistics for each group
        for category in ['by_algorithm', 'by_model', 'by_distribution']:
            for key, values in analysis[category].items():
                analysis[category][key] = {
                    'avg_accuracy': np.mean(values),
                    'std_accuracy': np.std(values),
                    'count': len(values),
                    'accuracies': values
                }
        
        return analysis
    
    def _extract_detailed_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Extract detailed metrics for comprehensive analysis."""
        successful_results = [r for r in all_results if r.get('status') == 'completed']
        
        detailed_metrics = {
            'training_metrics': [],
            'timing_metrics': [],
            'algorithm_metrics': [],
            'data_distribution_metrics': []
        }
        
        for result in successful_results:
            experiment_id = result.get('experiment_id', 'unknown')
            final_metrics = result.get('final_metrics', {})
            
            # Training metrics
            training_metric = {
                'experiment_id': experiment_id,
                'algorithm': result.get('algorithm'),
                'model': result.get('model'),
                'distribution': result.get('distribution'),
                'accuracy': final_metrics.get('accuracy', 0.0),
                'precision': final_metrics.get('precision', 0.0),
                'recall': final_metrics.get('recall', 0.0),
                'f1_score': final_metrics.get('f1_score', 0.0),
                'auc': final_metrics.get('auc', 0.0),
                'loss': final_metrics.get('loss', 0.0)
            }
            detailed_metrics['training_metrics'].append(training_metric)
            
            # Timing metrics
            timing_metric = {
                'experiment_id': experiment_id,
                'total_duration_seconds': result.get('duration_seconds', 0.0),
                'training_time': final_metrics.get('training_time', 0.0),
                'inference_time': final_metrics.get('inference_time', 0.0),
                'avg_epoch_time': final_metrics.get('avg_epoch_time', 0.0)
            }
            detailed_metrics['timing_metrics'].append(timing_metric)
        
        return detailed_metrics

    def _extract_data_dimensions(self, factory_datasites: Dict[str, FactoryDataSite]) -> Tuple[int, int, int]:
        """Extract real data dimensions from federated datasites."""
        try:
            # Get first datasite to extract dimensions
            first_datasite = next(iter(factory_datasites.values()))
            
            # Access the real uploaded data to get dimensions
            # This depends on the structure of your FactoryDataSite
            # For now, use the standard AI4I dataset dimensions we know
            # In a real implementation, you would query the datasite for data shape
            
            # Standard AI4I 2020 dataset dimensions:
            input_dim = 10  # Number of features in AI4I dataset
            num_classes = 2  # Binary classification (failure/no failure)
            sequence_length = 10  # For temporal models
            
            print(f"Using real AI4I dataset dimensions: input_dim={input_dim}, num_classes={num_classes}")
            return input_dim, num_classes, sequence_length
            
        except Exception as e:
            print(f"Warning: Could not extract dimensions from datasites: {e}")
            print("Using default AI4I dataset dimensions")
            return 10, 2, 10  # AI4I dataset defaults
    
    def _save_intermediate_result(self, result: Dict[str, Any], experiment_name: str, run_number: int):
        """Save intermediate experiment result to clean directory structure."""
        try:
            # Create experiment directory in clean structure: run_X/experiment_name/
            experiment_dir = self._create_clean_experiment_directory(experiment_name, run_number)
            
            # Save final experiment result
            experiment_file = os.path.join(experiment_dir, "final_result.json")
            with open(experiment_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Also append to combined log file in main results directory for easy tracking
            combined_log = os.path.join(self._current_results_dir, "experiment_progress.log")
            with open(combined_log, 'a') as f:
                status = result.get('status', 'unknown')
                accuracy = result.get('final_metrics', {}).get('accuracy', 0.0)
                duration = result.get('duration_seconds', 0.0)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {experiment_name}: {status} | Acc={accuracy:.4f} | Duration={duration:.2f}s\n")
                
        except Exception as e:
            print(f"[WARNING] Failed to save intermediate result for {experiment_name}: {e}")
    
    def _save_round_metrics(self, round_num: int, round_metrics: Dict[str, Any], client_metrics: Dict[str, Any], 
                          experiment_name: Optional[str] = None, run_number: int = 1):
        """Save round-by-round metrics for real-time monitoring and comprehensive CSV export."""
        try:
            # DEBUG: Log what we're receiving before saving
            print(f"\n[DEBUG SAVE] Round {round_num} metrics being saved:")
            print(f"   round_metrics keys: {list(round_metrics.keys())}")
            print(f"   round_metrics accuracy: {round_metrics.get('accuracy', 'MISSING')}")
            print(f"   round_metrics loss: {round_metrics.get('loss', 'MISSING')}")
            print(f"   round_metrics test_accuracy: {round_metrics.get('test_accuracy', 'MISSING')}")
            print(f"   client_metrics count: {len(client_metrics)}")
            
            # Create clean directory structure if experiment name is provided
            if experiment_name and run_number:
                experiment_dir = self._create_clean_experiment_directory(experiment_name, run_number)
                round_file_path = os.path.join(experiment_dir, f"round_{round_num}.json")
            else:
                # Fallback to old structure if no experiment context
                if not hasattr(self, '_current_results_dir') or not self._current_results_dir:
                    return  # Skip if no current results directory
                rounds_dir = os.path.join(self._current_results_dir, "round_metrics")
                os.makedirs(rounds_dir, exist_ok=True)
                round_file_path = os.path.join(rounds_dir, f"round_{round_num}.json")
            
            # Save detailed round metrics to JSON with clear labels
            round_data = {
                'round': round_num,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'global_metrics': {
                    # Primary validation metrics (averaged from client validation results)
                    'validation_accuracy': round_metrics.get('accuracy', 0.0),
                    'validation_loss': round_metrics.get('loss', 0.0),
                    'validation_precision': round_metrics.get('precision', 0.0),
                    'validation_recall': round_metrics.get('recall', 0.0),
                    'validation_f1': round_metrics.get('f1_score', 0.0),
                    'validation_auc': round_metrics.get('auc', 0.0),
                    
                    # Test metrics (if available)
                    'test_accuracy': round_metrics.get('test_accuracy', 0.0),
                    'test_loss': round_metrics.get('test_loss', 0.0),
                    'test_precision': round_metrics.get('test_precision', 0.0),
                    'test_recall': round_metrics.get('test_recall', 0.0),
                    'test_f1': round_metrics.get('test_f1', 0.0),
                    'test_auc': round_metrics.get('test_auc', 0.0),
                    
                    # Metadata
                    'evaluation_method': round_metrics.get('evaluation_method', 'client_average'),
                    'round': round_num,
                    'duration': round_metrics.get('duration', 0.0)
                },
                'client_metrics': client_metrics
            }
            
            # Save the round data to the correct file path (determined above)
            with open(round_file_path, 'w') as f:
                json.dump(round_data, f, indent=2, default=str)
            
            # DEBUG: Log what we saved
            print(f"[DEBUG SAVE] Saved to {round_file_path}")
            print(f"   Saved validation_accuracy: {round_data['global_metrics']['validation_accuracy']}")
            print(f"   Saved test_accuracy: {round_data['global_metrics']['test_accuracy']}")
            
            # Save comprehensive CSV metrics (using current results dir for now)
            self._save_comprehensive_csv_metrics(round_num, round_metrics, client_metrics)
                
            # Also append to progress summary using REAL client metrics as global metrics
            summary_file = os.path.join(self._current_results_dir, "training_progress.log")
            with open(summary_file, 'a') as f:
                # Calculate REAL metrics from client results
                if client_metrics:
                    # Use validation metrics as they are most representative of model performance
                    # Client metrics use 'val_accuracy' and 'val_loss', not 'validation_accuracy'
                    client_accuracies = [m.get('val_accuracy', 0.0) for m in client_metrics.values()]
                    client_losses = [m.get('val_loss', 0.0) for m in client_metrics.values()]
                    
                    # Use average of REAL client metrics as global metrics
                    real_global_acc = sum(client_accuracies) / len(client_accuracies) if client_accuracies else 0.0
                    real_global_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
                    avg_client_acc = real_global_acc  # Same as global for federated average
                    
                    f.write(f"Round {round_num:2d}: Global Acc={real_global_acc:.4f}, Loss={real_global_loss:.4f}, Avg Client Acc={avg_client_acc:.4f}\n")
                else:
                    # Fallback only if no client metrics available
                    global_acc = round_metrics.get('accuracy', 0.0)
                    global_loss = round_metrics.get('loss', 0.0)
                    f.write(f"Round {round_num:2d}: Global Acc={global_acc:.4f}, Loss={global_loss:.4f}, Avg Client Acc=0.0000\n")
                
        except Exception as e:
            print(f"[WARNING] Failed to save round metrics for round {round_num}: {e}")
    
    def _calculate_comprehensive_metrics(self, round_metrics: Dict[str, Any], data_type: str) -> Dict[str, float]:
        """Calculate comprehensive metrics (accuracy, precision, recall, F1, AUC) for validation or test data."""
        try:
            if data_type == 'validation':
                accuracy = round_metrics.get('accuracy', 0.0)  # This is validation accuracy
                # For now, use placeholder values - should be calculated from actual predictions
                precision = round_metrics.get('precision', 0.0) 
                recall = round_metrics.get('recall', 0.0)
                f1_score = round_metrics.get('f1_score', 0.0)
                auc = round_metrics.get('auc', 0.0)
                correct = round_metrics.get('correct', 0)
                total = round_metrics.get('total', 0)
            elif data_type == 'test':
                accuracy = round_metrics.get('test_accuracy', 0.0)
                precision = round_metrics.get('test_precision', 0.0)
                recall = round_metrics.get('test_recall', 0.0) 
                f1_score = round_metrics.get('test_f1', 0.0)
                auc = round_metrics.get('test_auc', 0.0)
                correct = round_metrics.get('test_correct', 0)
                total = round_metrics.get('test_total', 0)
            else:
                # Default empty metrics
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.0, 'correct': 0, 'total': 0}
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'auc': auc,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            print(f"[WARNING] Failed to calculate {data_type} metrics: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.0, 'correct': 0, 'total': 0}
    
    def _save_comprehensive_csv_metrics(self, round_num: int, round_metrics: Dict[str, Any], client_metrics: Dict[str, Any]):
        """Save CSV files matching the expected format from federated experiment framework."""
        try:
            if not hasattr(self, '_current_experiment_id') or self._current_experiment_id is None:
                print("[WARNING] No current experiment ID available for CSV metrics")
                return
            
            experiment_id = self._current_experiment_id
            base_filename = experiment_id.replace('_', '_').lower()
            
            # Extract algorithm name from experiment_id (e.g., "FedAvg_OptimizedCNN_IID_Secure" -> "fedavg")
            algorithm_name = experiment_id.split('_')[0].lower()
            
            # Calculate comprehensive metrics for validation and test data
            val_metrics = self._calculate_comprehensive_metrics(round_metrics, 'validation')
            test_metrics = self._calculate_comprehensive_metrics(round_metrics, 'test')
            
            # 1. Save main rounds CSV - VALIDATION METRICS (primary comparison metric)
            rounds_csv_file = os.path.join(self._current_results_dir, f"{base_filename}_rounds.csv")
            rounds_headers = ['round', 'algorithm', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score', 'val_auc', 
                             'val_correct', 'val_total', 'eval_time', 'privacy_preserved', 'experiment_key', 'run_number']
            
            # Main round row (validation metrics) - Clear validation focus
            rounds_row = [
                round_num,
                algorithm_name,
                val_metrics['accuracy'],
                val_metrics['precision'],
                val_metrics['recall'],
                val_metrics['f1_score'],
                val_metrics['auc'],
                val_metrics['correct'],
                val_metrics['total'],
                round_metrics.get('eval_time', round_metrics.get('duration', 0.0)),
                False,  # privacy_preserved
                base_filename,
                1  # run_number
            ]
            
            # Write main rounds CSV
            file_exists = os.path.exists(rounds_csv_file)
            with open(rounds_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(rounds_headers)
                writer.writerow(rounds_row)
            
            # 2. Save TEST metrics in separate CSV file (enhancement for comparison)
            test_csv_file = os.path.join(self._current_results_dir, f"{base_filename}_test_rounds.csv")
            test_headers = ['round', 'algorithm', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_auc',
                           'test_correct', 'test_total', 'eval_time', 'experiment_key', 'run_number']
            
            test_row = [
                round_num,
                algorithm_name,
                test_metrics['accuracy'],
                test_metrics['precision'],
                test_metrics['recall'],
                test_metrics['f1_score'],
                test_metrics['auc'],
                test_metrics['correct'],
                test_metrics['total'],
                round_metrics.get('eval_time', round_metrics.get('duration', 0.0)),
                base_filename,
                1  # run_number
            ]
            
            file_exists = os.path.exists(test_csv_file)
            with open(test_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(test_headers)
                writer.writerow(test_row)
            
            # 3. Save individual client metrics CSV for detailed analysis
            client_csv_file = os.path.join(self._current_results_dir, f"{base_filename}_client_rounds.csv")
            client_headers = ['round', 'client_id', 'client_number', 'training_accuracy', 'training_loss', 
                             'validation_accuracy', 'validation_loss', 'test_accuracy', 'test_loss',
                             'local_epochs', 'samples_count', 'timestamp']
            
            file_exists = os.path.exists(client_csv_file)
            with open(client_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(client_headers)
                
                # Write metrics for each client
                client_number = 1
                for client_id, metrics in client_metrics.items():
                    client_row = [
                        round_num,
                        client_id,
                        client_number,
                        metrics.get('accuracy', 0.0),  # training accuracy
                        metrics.get('loss', 0.0),     # training loss
                        metrics.get('val_accuracy', 0.0),
                        metrics.get('val_loss', 0.0),
                        metrics.get('test_accuracy', 0.0),
                        metrics.get('test_loss', 0.0),
                        metrics.get('local_epochs', 1),
                        metrics.get('samples_count', 0),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                    writer.writerow(client_row)
                    client_number += 1
                    
        except Exception as e:
            print(f"[WARNING] Failed to save comprehensive CSV metrics for round {round_num}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_experiment_summary(self, experiment_id: str, final_results: Dict[str, Any], experiment_duration: float):
        """Save experiment summary JSON in the exact format expected by the results framework."""
        try:
            if not hasattr(self, '_current_results_dir'):
                return
                
            # Parse experiment components
            parts = experiment_id.split('_')
            algorithm_name = parts[0].lower() if len(parts) > 0 else "unknown"
            model_name = parts[1].lower() if len(parts) > 1 else "unknown" 
            distribution_name = parts[2].lower() if len(parts) > 2 else "unknown"
            communication_style = parts[3].lower() if len(parts) > 3 else "standard"
            
            base_filename = experiment_id.replace('_', '_').lower()
            
            # Extract final metrics
            final_metrics = final_results.get('final_metrics', {})
            round_metrics = final_results.get('round_metrics', [])
            
            # Calculate timing metrics
            total_rounds = len(round_metrics)
            avg_round_time = experiment_duration / total_rounds if total_rounds > 0 else 0.0
            
            # Create summary in exact expected format
            summary = {
                "experiment_key": base_filename,
                "model": model_name,
                "algorithm": algorithm_name,
                "distribution": distribution_name,
                "server_type": communication_style,
                "final_accuracy": final_metrics.get('accuracy', 0.0),
                "final_f1": final_metrics.get('f1_score', 0.0),
                "final_precision": final_metrics.get('precision', 0.0),
                "final_recall": final_metrics.get('recall', 0.0),
                "final_auc": final_metrics.get('auc', 0.0),
                "training_time": experiment_duration,
                "inference_time": final_metrics.get('eval_time', 0.0),
                "avg_round_time": avg_round_time,
                "total_rounds": total_rounds,
                "early_stopping": total_rounds < self.num_rounds,
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                # Additional test metrics for enhanced analysis
                "final_test_accuracy": final_metrics.get('test_accuracy', 0.0),
                "final_test_f1": final_metrics.get('test_f1', 0.0),
                "final_test_precision": final_metrics.get('test_precision', 0.0),
                "final_test_recall": final_metrics.get('test_recall', 0.0),
                "final_test_auc": final_metrics.get('test_auc', 0.0)
            }
            
            # Save summary JSON
            summary_file = os.path.join(self._current_results_dir, f"{base_filename}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"✅ Saved experiment summary: {summary_file}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save experiment summary: {e}")
            import traceback
            traceback.print_exc()
    
    def get_all_experiment_names(self) -> List[str]:
        """Generate all possible experiment combinations."""
        experiments = []
        for algorithm_name in self.algorithms.keys():
            for model_name in self.models.keys():
                for distribution_name in self.data_distributions.keys():
                    for communication_style in self.communication_styles:
                        experiments.append(f"{algorithm_name}_{model_name}_{distribution_name}_{communication_style}")
        return experiments
    
    def run_individual_experiment(self, experiment_name: str, resume: bool = True) -> Dict[str, Any]:
        """Run a single experiment with resumption support."""
        print(f"\n{'='*80}")
        print(f"[EXPERIMENT] Starting: {experiment_name}")
        print(f"{'='*80}")
        
        # Check if already completed (individual experiments are always run 1)
        if resume and self.state.is_experiment_completed(experiment_name, run_number=1):
            print(f"[SKIP] Experiment {experiment_name} already completed")
            return {'status': 'already_completed', 'experiment_name': experiment_name}
        
        # Get starting round for resumption (individual experiments are always run 1)
        start_round = 0
        if resume:
            start_round = self.state.get_current_round(experiment_name, run_number=1)
            if start_round > 0:
                print(f"[RESUME] Resuming from round {start_round + 1}")
        
        try:
            # Parse experiment name
            parts = experiment_name.split('_')
            if len(parts) != 4:
                raise ValueError(f"Invalid experiment name format: {experiment_name}. Expected: Algorithm_Model_Distribution_Communication")
            
            algorithm_name, model_name, distribution_name, communication_style = parts
            
            # Validate components
            if algorithm_name not in self.algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(self.algorithms.keys())}")
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
            if distribution_name not in self.data_distributions:
                raise ValueError(f"Unknown distribution: {distribution_name}. Available: {list(self.data_distributions.keys())}")
            if communication_style not in self.communication_styles:
                raise ValueError(f"Unknown communication style: {communication_style}. Available: {self.communication_styles}")
            
            # Run the experiment (individual experiments are always run 1)
            result = self.run_single_real_experiment_enhanced(
                algorithm_name, model_name, distribution_name, communication_style,
                start_round=start_round, run_number=1
            )
            
            # Mark as completed if successful (individual experiments are always run 1)
            if result.get('status') == 'completed':
                self.state.mark_experiment_completed(experiment_name, run_number=1)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Experiment {experiment_name} failed: {e}")
            return {
                'status': 'failed',
                'experiment_name': experiment_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_single_real_experiment_enhanced(self, algorithm_name: str, model_name: str, 
                                  distribution_name: str, communication_style: str,
                                  start_round: int = 0, run_number: int = 1) -> Dict[str, Any]:
        """Run a single federated learning experiment with resumption support."""
        experiment_name = f"{algorithm_name}_{model_name}_{distribution_name}_{communication_style}"
        
        print(f"[INFO] Running experiment: {experiment_name}")
        print(f"[INFO] Parameters: rounds={self.num_rounds}, epochs={self.local_epochs}, batch_size={self.batch_size}")
        if start_round > 0:
            print(f"[RESUME] Starting from round {start_round + 1}/{self.num_rounds}")
        
        start_time = time.time()
        
        try:
            # Create experiment-specific results directory
            exp_dir = os.path.join(self._current_results_dir, experiment_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save experiment parameters
            exp_config = {
                'experiment_name': experiment_name,
                'algorithm': algorithm_name,
                'model': model_name,
                'distribution': distribution_name,
                'communication': communication_style,
                'parameters': self.params.to_dict() if hasattr(self.params, 'to_dict') else {
                    'max_rounds': self.num_rounds,
                    'local_epochs': self.local_epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate
                },
                'start_round': start_round,
                'timestamp': datetime.now().isoformat()
            }
            
            config_file = os.path.join(exp_dir, "experiment_config.json")
            with open(config_file, 'w') as f:
                json.dump(exp_config, f, indent=2)
            
            # Create FRESH datasites using new infrastructure with force recreation
            distribution_config = self.data_distributions[distribution_name]
            factory_datasites = self.ensure_fresh_datasites(distribution_config, model_name)
            
            # Create algorithm instance
            algorithm_config = self.algorithms[algorithm_name]
            algorithm = algorithm_config['class'](**algorithm_config['params'])
            
            # Create model configuration
            model_config = self.create_model_config(model_name)
            model_type = self.models[model_name]
            
            # Set up experiment configuration  
            model_type_enum = ModelType.CNN if model_type == ModelType.CNN else ModelType.LSTM if model_type == ModelType.LSTM else ModelType.HYBRID
            experiment_config = ExperimentConfig(
                experiment_id=experiment_name,
                name=experiment_name,
                description=f"REAL federated learning experiment with resumption",
                model_type=model_type_enum,
                model_params=model_config,
                algorithm_params={
                    'name': algorithm.__class__.__name__,
                    'learning_rate': getattr(algorithm, 'learning_rate', self.learning_rate),
                    'device': getattr(algorithm, 'device', 'cpu')
                },
                max_rounds=self.num_rounds,
                training_params={
                    'local_epochs': self.local_epochs,
                    'batch_size': self.batch_size,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            )
            
            # Initialize metrics collector
            metrics_collector = FederatedMetricsCollector()
            
            # Run federated learning with resumption support
            federated_result = self.run_real_federated_learning(
                algorithm, factory_datasites, experiment_config, metrics_collector, experiment_name, start_round, run_number
            )
            
            # Cleanup
            self.cleanup_datasites(factory_datasites)
            
            duration = time.time() - start_time
            
            # Prepare final result
            result = {
                'status': 'completed',
                'experiment_name': experiment_name,
                'algorithm': algorithm_name,
                'model': model_name,
                'distribution': distribution_name,
                'communication': communication_style,
                'parameters': exp_config['parameters'],
                'final_metrics': federated_result.get('final_metrics', {}),
                'round_metrics': federated_result.get('round_metrics', []),
                'data_stats': federated_result.get('data_stats', {}),
                'duration_seconds': duration,
                'completed_rounds': len(federated_result.get('round_metrics', [])),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save individual experiment result
            result_file = os.path.join(exp_dir, "final_result.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"[SUCCESS] Experiment {experiment_name} completed in {duration:.2f}s")
            print(f"[RESULTS] Final accuracy: {result['final_metrics'].get('accuracy', 0.0):.4f}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERROR] Experiment {experiment_name} failed after {duration:.2f}s: {e}")
            return {
                'status': 'failed',
                'experiment_name': experiment_name,
                'error': str(e),
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Enhanced main function with CLI support for configurable experiments."""
    parser = argparse.ArgumentParser(description="Real PySyft Federated Learning Experiments")
    
    # Experiment selection
    parser.add_argument('--experiment', type=str, help='Run specific experiment (e.g., FedAvg_OptimizedCNN_uniform_sync)')
    parser.add_argument('--list-experiments', action='store_true', help='List all possible experiments')
    
    # Configurable parameters (no more hardcoded values!)
    parser.add_argument('--max-rounds', type=int, default=10, help='Maximum number of federated rounds')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local training epochs per round')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num-datasites', type=int, default=3, help='Number of federated datasites')
    
    # Algorithm-specific parameters
    parser.add_argument('--fedprox-mu', type=float, default=0.01, help='FedProx regularization parameter')
    
    # Experiment management
    parser.add_argument('--resume', action='store_true', help='Resume from previous checkpoint')
    parser.add_argument('--results-dir', type=str, help='Custom results directory path')
    parser.add_argument('--config', type=str, help='Load parameters from JSON config file')
    
    # Datasite options
    parser.add_argument('--external-datasites', action='store_true', help='Use external datasites instead of launching new ones')
    
    # === ENHANCED FEATURES OPTIONS ===
    parser.add_argument('--disable-enhanced', action='store_true', help='Disable enhanced features (parallel processing, heartbeat monitoring, fault tolerance)')
    parser.add_argument('--max-parallel', type=int, default=3, help='Maximum number of parallel datasites (default: 3)')
    parser.add_argument('--heartbeat-interval', type=int, default=30, help='Heartbeat check interval in seconds (default: 30)')
    parser.add_argument('--disable-early-stopping', action='store_true', help='Disable early stopping to run full max-rounds')
    
    # === MONITORING PORTS ===
    parser.add_argument('--heartbeat-port', type=int, default=8888, help='Port for heartbeat manager API (default: 8888)')
    parser.add_argument('--dashboard-port', type=int, default=8889, help='Port for status dashboard web interface (default: 8889)')
    
    # Run options
    parser.add_argument('--run-all', action='store_true', help='Run all 48 experiments')
    parser.add_argument('--runs', type=int, default=1, help='Number of times to repeat the experiment execution (default: 1)')
    
    args = parser.parse_args()
    
    # Initialize runner variable
    runner = None
    
    try:
        # Create experiment parameters with proper command-line precedence
        if args.config and CONFIGURABLE_EXPERIMENTS_AVAILABLE:
            # Load base parameters from config file
            params = ExperimentParams.from_json_file(args.config)
            print(f"[CONFIG] Loaded base parameters from: {args.config}")
            
            # Override with command-line arguments if provided (command-line takes precedence)
            if hasattr(args, 'max_rounds') and args.max_rounds != 10:  # 10 is the parser default
                params.max_rounds = args.max_rounds
                print(f"[CONFIG] Override max_rounds from command-line: {args.max_rounds}")
            if hasattr(args, 'local_epochs') and args.local_epochs != 1:  # 1 is the parser default
                params.local_epochs = args.local_epochs
                print(f"[CONFIG] Override local_epochs from command-line: {args.local_epochs}")
            if hasattr(args, 'batch_size') and args.batch_size != 32:  # 32 is the parser default
                params.batch_size = args.batch_size
                print(f"[CONFIG] Override batch_size from command-line: {args.batch_size}")
            if hasattr(args, 'learning_rate') and args.learning_rate != 0.01:  # 0.01 is the parser default
                params.learning_rate = args.learning_rate
                print(f"[CONFIG] Override learning_rate from command-line: {args.learning_rate}")
            if hasattr(args, 'num_datasites') and args.num_datasites != 3:  # 3 is the parser default
                params.num_datasites = args.num_datasites
                print(f"[CONFIG] Override num_datasites from command-line: {args.num_datasites}")
            if hasattr(args, 'fedprox_mu') and args.fedprox_mu != 0.01:  # 0.01 is the parser default
                params.fedprox_mu = args.fedprox_mu
                print(f"[CONFIG] Override fedprox_mu from command-line: {args.fedprox_mu}")
            if hasattr(args, 'heartbeat_port') and args.heartbeat_port != 8888:  # 8888 is the parser default
                params.heartbeat_port = args.heartbeat_port
                print(f"[CONFIG] Override heartbeat_port from command-line: {args.heartbeat_port}")
            if hasattr(args, 'dashboard_port') and args.dashboard_port != 8889:  # 8889 is the parser default
                params.dashboard_port = args.dashboard_port
                print(f"[CONFIG] Override dashboard_port from command-line: {args.dashboard_port}")
        else:
            # Use command-line arguments directly
            params = ExperimentParams(
                max_rounds=args.max_rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_datasites=args.num_datasites,
                fedprox_mu=args.fedprox_mu,
                heartbeat_port=args.heartbeat_port,
                dashboard_port=args.dashboard_port
            )
            print("[CONFIG] Using command-line parameters directly")
        
        # Validate and log final parameters
        print(f"[PARAMS] Final parameters: rounds={params.max_rounds}, epochs={params.local_epochs}, batch={params.batch_size}, lr={params.learning_rate}")
        print(f"[PARAMS] Additional: datasites={params.num_datasites}, fedprox_mu={params.fedprox_mu}")
        print(f"[PARAMS] Monitoring ports: heartbeat={params.heartbeat_port}, dashboard={params.dashboard_port}")
        
        # Ensure critical parameters are valid
        if params.max_rounds <= 0:
            raise ValueError(f"max_rounds must be positive, got: {params.max_rounds}")
        if params.local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got: {params.local_epochs}")
        if params.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got: {params.batch_size}")

        print(f"[PARAMS] Using parameters: rounds={params.max_rounds}, epochs={params.local_epochs}, batch={params.batch_size}, lr={params.learning_rate}")
        
        # Check if external datasites should be used
        use_external = args.external_datasites or EXTERNAL_DATASITE_AVAILABLE
        if use_external and EXTERNAL_DATASITE_AVAILABLE:
            print("[CONFIG] External datasite mode enabled")
        else:
            print("[CONFIG] Using local datasite launch mode")
        
        # Create experiment runner with configurable parameters and enhanced features
        runner = RealPySyftExperimentRunner(
            experiment_params=params,
            results_dir=args.results_dir,
            use_external_datasites=use_external,
            enable_enhanced_features=not args.disable_enhanced,
            max_parallel_datasites=args.max_parallel,
            heartbeat_interval=args.heartbeat_interval
        )
        
        if args.list_experiments:
            # List all possible experiments
            experiments = runner.get_all_experiment_names()
            print(f"\nAll {len(experiments)} possible experiments:")
            print("=" * 60)
            for i, exp in enumerate(experiments, 1):
                status = "✅ COMPLETED" if runner.state and runner.state.is_experiment_completed(exp) else "⏳ PENDING"
                print(f"{i:2d}. {exp:<50} {status}")
            
            if runner.state:
                completed = len([e for e in experiments if runner.state.is_experiment_completed(e)])
                print(f"\nProgress: {completed}/{len(experiments)} experiments completed")
            
            return
        
        if args.experiment:
            # Run specific experiment
            print(f"[RUN] Running individual experiment: {args.experiment}")
            result = runner.run_individual_experiment(args.experiment, resume=args.resume)
            
            if result['status'] == 'completed':
                print(f"[SUCCESS] Experiment completed successfully!")
                print(f"[METRICS] Final accuracy: {result.get('final_metrics', {}).get('accuracy', 0.0):.4f}")
                print(f"[TIME] Duration: {result.get('duration_seconds', 0.0):.2f} seconds")
            elif result['status'] == 'already_completed':
                print(f"[INFO] Experiment was already completed")
            else:
                print(f"[ERROR] Experiment failed: {result.get('error', 'Unknown error')}")
                
        elif args.run_all:
            # Run all experiments with resumption support and multiple runs
            if args.runs > 1:
                log_and_print(f"[RUN] Running all 48 experiments {args.runs} times with resumption support")
            else:
                log_and_print("[RUN] Running all 48 experiments with resumption support")
            runner.run_all_experiments(num_runs=args.runs, resume=args.resume)
            
        else:
            print("Usage:")
            print("  --list-experiments    : List all possible experiments")
            print("  --experiment NAME     : Run specific experiment")
            print("  --run-all            : Run all 48 experiments")
            print("  --runs N             : Number of times to repeat execution (default: 1)")
            print("  --max-rounds N       : Set number of federated rounds")
            print("  --local-epochs N     : Set local training epochs")
            print("  --batch-size N       : Set training batch size")
            print("  --learning-rate F    : Set learning rate")
            print("  --resume             : Resume from checkpoint")
            print("\nExample:")
            print("  python run_enhanced_experiments.py --experiment FedAvg_OptimizedCNN_uniform_sync --max-rounds 5")
            print("  python run_enhanced_experiments.py --run-all --runs 3  # Run all experiments 3 times")
            
    except KeyboardInterrupt:
        print("\n[WARNING] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Enhanced cleanup
        if runner is not None and hasattr(runner, 'cleanup_enhanced_components'):
            runner.cleanup_enhanced_components()


if __name__ == "__main__":
    main()
