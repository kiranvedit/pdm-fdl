"""
Experiment Logger for Federated Learning
Real-time logging and status tracking with enhanced table displays
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from io import StringIO
import sys


class TeeLogger:
    """
    Captures all console output and writes to both console and log file
    """
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file
        
    def write(self, text):
        # Write to original stream (console)
        self.original_stream.write(text)
        self.original_stream.flush()
        
        # Write to log file with timestamp
        if self.log_file and not self.log_file.closed:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if text.strip():  # Only add timestamp to non-empty lines
                self.log_file.write(f"[{timestamp}] {text}")
            else:
                self.log_file.write(text)
            self.log_file.flush()
    
    def flush(self):
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()


class ExperimentLogger:
    """
    Comprehensive logging system for federated learning experiments
    Provides real-time status tables and detailed experiment tracking
    """
    
    def __init__(self, log_dir: str = "logs", enable_file_logging: bool = True, 
                 enable_console_logging: bool = True, log_level: str = "INFO",
                 capture_console_output: bool = True):
        """
        Initialize experiment logger
        
        Args:
            log_dir: Directory for log files
            enable_file_logging: Whether to write logs to files
            enable_console_logging: Whether to display logs in console
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            capture_console_output: Whether to capture all console output to logs
        """
        self.log_dir = log_dir
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.capture_console_output = capture_console_output
        
        # Create log directory
        if enable_file_logging and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Current experiment tracking
        self.current_session = None
        self.current_run = None
        self.current_experiment = None
        self.experiment_start_time = None
        self.run_start_time = None
        
        # Status tracking for real-time display
        self.experiment_status = []
        self.round_status = []
        self.last_status_display = 0
        self.status_display_interval = 5  # seconds
        
        # Console output capturing
        self.detailed_log_file = None
        self.original_stdout = None
        
        # Setup logging
        self._setup_logging(log_level)
        
    def _setup_logging(self, log_level: str):
        """Setup logging configuration"""
        # Configure logging level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(f"federated_experiment_{id(self)}")
        self.logger.setLevel(numeric_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (will be created per session)
        self.file_handler = None
        
    def start_session(self, session_name: str):
        """Start a new experiment session"""
        self.current_session = session_name
        self.experiment_status = []
        self.round_status = []
        
        # Setup file logging for this session
        if self.enable_file_logging:
            log_file = os.path.join(self.log_dir, f"{session_name}.log")
            self.file_handler = logging.FileHandler(log_file, mode='w')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            self.file_handler.setFormatter(file_formatter)
            self.logger.addHandler(self.file_handler)
            
            # Setup detailed console output capture
            if self.capture_console_output:
                detailed_log_file = os.path.join(self.log_dir, f"{session_name}_detailed.log")
                self.detailed_log_file = open(detailed_log_file, 'w', encoding='utf-8')
                
                # Write session header to detailed log
                self.detailed_log_file.write(f"{'='*100}\n")
                self.detailed_log_file.write(f"FEDERATED LEARNING EXPERIMENT SESSION: {session_name}\n")
                self.detailed_log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.detailed_log_file.write(f"{'='*100}\n\n")
                self.detailed_log_file.flush()
                
                # Redirect stdout to capture all console output
                self.original_stdout = sys.stdout
                sys.stdout = TeeLogger(self.original_stdout, self.detailed_log_file)
        
        self.logger.info(f"Started experiment session: {session_name}")
        
        print(f"\n🔍 LOGGING ENABLED")
        print(f"📁 Session: {session_name}")
        print(f"📊 Status tables: Every {self.status_display_interval} seconds")
        print(f"📝 Detailed logs: logs/{session_name}_detailed.log")
        print(f"📈 Structured logs: logs/{session_name}.log")
        print(f"{'='*60}")
        
    def start_run(self, run_number: int, total_runs: int):
        """Start a new run within the session"""
        self.current_run = run_number
        self.run_start_time = time.time()
        
        self.logger.info(f"Started run {run_number}/{total_runs}")
        print(f"\n{'='*80}")
        print(f"RUN {run_number}/{total_runs} - {self.current_session}")
        print(f"{'='*80}")
        
    def start_experiment(self, experiment_key: str, config: Dict):
        """Start a new experiment within a run"""
        self.current_experiment = experiment_key
        self.experiment_start_time = time.time()
        self.round_status = []
        
        # Parse experiment configuration
        parts = experiment_key.split('_')
        model = parts[0] if len(parts) > 0 else "unknown"
        algorithm = parts[1] if len(parts) > 1 else "unknown"
        distribution = parts[2] if len(parts) > 2 else "unknown"
        server_type = parts[3] if len(parts) > 3 else "unknown"
        
        self.logger.info(f"Started experiment: {experiment_key}")
        self.logger.info(f"Config: {config}")
        
        # Add to experiment status tracking
        experiment_info = {
            'run': self.current_run,
            'experiment': experiment_key,
            'model': model,
            'algorithm': algorithm,
            'distribution': distribution,
            'server_type': server_type,
            'start_time': datetime.now(),
            'status': 'RUNNING',
            'current_round': 0,
            'max_rounds': config.get('max_rounds', 0),
            'current_accuracy': 0.0,
            'current_f1': 0.0,
            'current_precision': 0.0,
            'current_recall': 0.0,
            'current_auc': 0.0,
            'training_time': 0.0,
            'inference_time': 0.0,
            'error': None
        }
        self.experiment_status.append(experiment_info)
        
        print(f"\n📊 EXPERIMENT: {experiment_key}")
        print(f"   Model: {model} | Algorithm: {algorithm} | Distribution: {distribution} | Server: {server_type}")
        
    def log_round_start(self, round_num: int, max_rounds: int):
        """Log the start of a training round"""
        self.logger.debug(f"Round {round_num}/{max_rounds} started")
        
        # Update current experiment status
        if self.experiment_status:
            self.experiment_status[-1]['current_round'] = round_num
            
    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Log metrics for a completed round"""
        self.logger.info(f"Round {round_num} metrics: {metrics}")
        
        # Store round status
        round_info = {
            'run': self.current_run,
            'experiment': self.current_experiment,
            'round': round_num,
            'timestamp': datetime.now(),
            **metrics
        }
        self.round_status.append(round_info)
        
        # Update current experiment status
        if self.experiment_status:
            exp_status = self.experiment_status[-1]
            exp_status['current_round'] = round_num
            exp_status['current_accuracy'] = metrics.get('accuracy', 0.0)
            exp_status['current_f1'] = metrics.get('f1_score', 0.0)
            exp_status['current_precision'] = metrics.get('precision', 0.0)
            exp_status['current_recall'] = metrics.get('recall', 0.0)
            exp_status['current_auc'] = metrics.get('auc', 0.0)
            
        # Log detailed round information
        self.log_verbose(f"Round {round_num} completed:")
        self.log_verbose(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        self.log_verbose(f"  F1-Score: {metrics.get('f1_score', 0.0):.4f}")
        self.log_verbose(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        self.log_verbose(f"  Recall: {metrics.get('recall', 0.0):.4f}")
        self.log_verbose(f"  AUC: {metrics.get('auc', 0.0):.4f}")
        
        # Display status table periodically
        current_time = time.time()
        if current_time - self.last_status_display > self.status_display_interval:
            self.display_current_status()
            self.last_status_display = current_time
    
    def log_verbose(self, message: str, level: str = "INFO"):
        """Log verbose messages to detailed log file"""
        if self.detailed_log_file and not self.detailed_log_file.closed:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.detailed_log_file.write(f"[{timestamp}] {level}: {message}\n")
            self.detailed_log_file.flush()
            
    def log_client_update(self, client_id: str, metrics: Dict):
        """Log client update information"""
        self.log_verbose(f"Client {client_id} update: {metrics}")
        
    def log_aggregation_start(self, num_updates: int, algorithm: str):
        """Log aggregation start"""
        self.log_verbose(f"Starting {algorithm.upper()} aggregation with {num_updates} client updates")
        
    def log_aggregation_complete(self, aggregation_time: float, round_time: float):
        """Log aggregation completion"""
        self.log_verbose(f"Aggregation complete (Aggregation: {aggregation_time:.2f}s, Round: {round_time:.2f}s)")
        
    def log_early_stopping(self, round_num: int, reason: str):
        """Log early stopping trigger"""
        self.logger.warning(f"Early stopping triggered at round {round_num}: {reason}")
        self.log_verbose(f"EARLY STOPPING: Round {round_num} - {reason}")
        
    def log_model_evaluation(self, eval_time: float, test_samples: int):
        """Log model evaluation details"""
        self.log_verbose(f"Global model evaluation completed in {eval_time:.4f}s on {test_samples} test samples")
            
    def log_experiment_complete(self, experiment_key: str, result: Dict):
        """Log completion of an experiment"""
        duration = time.time() - self.experiment_start_time if self.experiment_start_time else 0
        
        if 'error' in result:
            self.logger.error(f"Experiment {experiment_key} FAILED: {result['error']}")
            status = 'FAILED'
            error = result['error']
        else:
            self.logger.info(f"Experiment {experiment_key} COMPLETED in {duration:.2f}s")
            status = 'SUCCESS'
            error = None
            
        # Update experiment status
        if self.experiment_status:
            exp_status = self.experiment_status[-1]
            exp_status['status'] = status
            exp_status['training_time'] = duration
            exp_status['error'] = error
            
            # Extract final metrics
            if 'experiment_summary' in result:
                summary = result['experiment_summary']
                exp_status['final_accuracy'] = summary.get('final_accuracy', 0.0)
                exp_status['training_time'] = summary.get('total_training_time', duration)
                
    def log_experiment_error(self, experiment_key: str, error: str):
        """Log an experiment error"""
        self.logger.error(f"Experiment {experiment_key} ERROR: {error}")
        
        # Update experiment status
        if self.experiment_status:
            exp_status = self.experiment_status[-1]
            exp_status['status'] = 'FAILED'
            exp_status['error'] = error
            
    def display_current_status(self):
        """Display current experiment status table"""
        if not self.experiment_status:
            return
            
        print(f"\n🔄 LIVE STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("="*120)
        
        # Create status table
        status_data = []
        for exp in self.experiment_status[-10:]:  # Show last 10 experiments
            status_data.append({
                'Run': exp['run'],
                'Experiment': exp['experiment'][:25] + '...' if len(exp['experiment']) > 25 else exp['experiment'],
                'Status': exp['status'],
                'Round': f"{exp['current_round']}/{exp['max_rounds']}",
                'Acc': f"{exp['current_accuracy']:.3f}",
                'F1': f"{exp['current_f1']:.3f}",
                'Prec': f"{exp['current_precision']:.3f}",
                'Rec': f"{exp['current_recall']:.3f}",
                'AUC': f"{exp['current_auc']:.3f}",
                'Time': f"{exp['training_time']:.1f}s",
                'Error': exp['error'][:20] + '...' if exp['error'] and len(exp['error']) > 20 else exp['error']
            })
            
        if status_data:
            df = pd.DataFrame(status_data)
            print(df.to_string(index=False, max_cols=12))
        
        print("="*120)
        
    def display_run_summary(self, run_number: int, run_results: Dict):
        """Display summary for completed run"""
        if not self.run_start_time:
            return
            
        run_duration = time.time() - self.run_start_time
        
        # Count successes and failures
        successes = sum(1 for exp in self.experiment_status if exp['run'] == run_number and exp['status'] == 'SUCCESS')
        failures = sum(1 for exp in self.experiment_status if exp['run'] == run_number and exp['status'] == 'FAILED')
        total = successes + failures
        
        print(f"\n📋 RUN {run_number} SUMMARY")
        print("="*60)
        print(f"Duration: {run_duration/60:.1f} minutes")
        print(f"Experiments: {successes}/{total} successful ({successes/total*100:.1f}%)")
        
        if failures > 0:
            print(f"\n❌ Failed Experiments:")
            for exp in self.experiment_status:
                if exp['run'] == run_number and exp['status'] == 'FAILED':
                    print(f"   {exp['experiment']}: {exp['error']}")
                    
        print("="*60)
        
    def display_session_summary(self):
        """Display final session summary"""
        if not self.experiment_status:
            return
            
        print(f"\n📊 SESSION SUMMARY - {self.current_session}")
        print("="*80)
        
        total_experiments = len(self.experiment_status)
        successful = sum(1 for exp in self.experiment_status if exp['status'] == 'SUCCESS')
        failed = sum(1 for exp in self.experiment_status if exp['status'] == 'FAILED')
        
        print(f"Total experiments: {total_experiments}")
        print(f"Successful: {successful} ({successful/total_experiments*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total_experiments*100:.1f}%)")
        
        # Show failed experiments
        if failed > 0:
            print(f"\n❌ Failed Experiments:")
            failure_counts = {}
            for exp in self.experiment_status:
                if exp['status'] == 'FAILED':
                    error = exp['error'] or 'Unknown error'
                    failure_counts[error] = failure_counts.get(error, 0) + 1
                    
            for error, count in failure_counts.items():
                print(f"   {error}: {count} experiments")
                
        print("="*80)
        
    def close_session(self):
        """Close the current session and cleanup"""
        if self.current_session:
            self.display_session_summary()
            self.logger.info(f"Closed experiment session: {self.current_session}")
            
        # Restore original stdout
        if self.original_stdout:
            sys.stdout = self.original_stdout
            self.original_stdout = None
            
        # Close detailed log file
        if self.detailed_log_file and not self.detailed_log_file.closed:
            self.detailed_log_file.write(f"\n{'='*100}\n")
            self.detailed_log_file.write(f"Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.detailed_log_file.write(f"{'='*100}\n")
            self.detailed_log_file.close()
            self.detailed_log_file = None
            
        # Remove file handler
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
            
        # Reset state
        self.current_session = None
        self.current_run = None
        self.current_experiment = None
        self.experiment_status = []
        self.round_status = []


# Global logger instance
_global_logger = None

def get_experiment_logger() -> ExperimentLogger:
    """Get global experiment logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger()
    return _global_logger

def setup_experiment_logging(log_dir: str = "logs", log_level: str = "INFO") -> ExperimentLogger:
    """Setup global experiment logging"""
    global _global_logger
    _global_logger = ExperimentLogger(log_dir=log_dir, log_level=log_level)
    return _global_logger
