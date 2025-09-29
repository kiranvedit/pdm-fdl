"""
Comprehensive Experiment Logging and Results Storage System
Handles detailed logging, metrics storage, and early stopping tracking for federated learning experiments.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class TeeLogger:
    """Captures output to both console and file."""
    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


class ExperimentLogger:
    """Comprehensive experiment logging and results storage system."""
    
    def __init__(self, experiment_id: str, results_base_dir: str = "results"):
        self.experiment_id = experiment_id
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create experiment-specific directories
        self.experiment_dir = Path(results_base_dir) / f"experiment_{experiment_id}_{self.timestamp}"
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics" 
        self.raw_data_dir = self.experiment_dir / "raw_data"
        
        # Create directories
        for dir_path in [self.experiment_dir, self.logs_dir, self.metrics_dir, self.raw_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.log_file = self.logs_dir / f"{experiment_id}_detailed.log"
        self._setup_logger()
        
        # Initialize storage structures
        self.experiment_metadata: Dict[str, Any] = {
            'experiment_id': experiment_id,
            'start_time': self.start_time.isoformat(),
            'status': 'running'
        }
        
        self.round_metrics = []
        self.client_metrics = {}
        self.global_metrics = []
        self.early_stopping_events = []
        self.error_log = []
        
        self.logger.info(f"≡ƒÜÇ Started experiment {experiment_id}")
        self.logger.info(f"≡ƒôü Results directory: {self.experiment_dir}")
    
    def _setup_logger(self):
        """Setup detailed file logging."""
        # Create logger
        self.logger = logging.getLogger(f"experiment_{self.experiment_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for key info only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment configuration and setup."""
        self.experiment_metadata.update({
            'config': config,
            'num_rounds': config.get('max_rounds', 0),
            'num_clients': config.get('num_clients', 0),
            'algorithm': config.get('algorithm_params', {}).get('name', 'unknown'),
            'model_type': str(config.get('model_type', 'unknown')),
            'distribution': config.get('distribution_params', {}),
        })
        
        self.logger.info("="*80)
        self.logger.info(f"EXPERIMENT CONFIGURATION")
        self.logger.info("="*80)
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Algorithm: {self.experiment_metadata['algorithm']}")
        self.logger.info(f"Model Type: {self.experiment_metadata['model_type']}")
        self.logger.info(f"Number of Rounds: {self.experiment_metadata['num_rounds']}")
        self.logger.info(f"Number of Clients: {self.experiment_metadata['num_clients']}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
        self.logger.info("="*80)
    
    def log_round_start(self, round_num: int, total_rounds: int):
        """Log the start of a training round."""
        self.logger.info(f"\n≡ƒôì ROUND {round_num + 1}/{total_rounds} - Started")
        print(f"\n≡ƒôì Round {round_num + 1}/{total_rounds}")  # Console display
    
    def log_client_training(self, client_id: str, round_num: int, metrics: Dict[str, Any]):
        """Log detailed client training metrics."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        
        client_round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy()
        }
        self.client_metrics[client_id].append(client_round_data)
        
        # Detailed logging
        self.logger.info(f"≡ƒöº Client {client_id} Training Results:")
        
        # Safe formatting for numeric values
        training_loss = metrics.get('training_loss', 'N/A')
        if isinstance(training_loss, (int, float)):
            self.logger.info(f"   Training Loss: {training_loss:.6f}")
        else:
            self.logger.info(f"   Training Loss: {training_loss}")
            
        # Training accuracy (NEW)
        training_accuracy = metrics.get('training_accuracy', metrics.get('train_accuracy', 'N/A'))
        if isinstance(training_accuracy, (int, float)):
            self.logger.info(f"   Training Accuracy: {training_accuracy:.6f}")
        else:
            self.logger.info(f"   Training Accuracy: {training_accuracy}")
            
        val_loss = metrics.get('val_loss', 'N/A')
        if isinstance(val_loss, (int, float)):
            self.logger.info(f"   Validation Loss: {val_loss:.6f}")
        else:
            self.logger.info(f"   Validation Loss: {val_loss}")
            
        val_accuracy = metrics.get('val_accuracy', 'N/A')
        if isinstance(val_accuracy, (int, float)):
            self.logger.info(f"   Validation Accuracy: {val_accuracy:.6f}")
        else:
            self.logger.info(f"   Validation Accuracy: {val_accuracy}")
            
        num_samples = metrics.get('num_samples', 'N/A')
        self.logger.info(f"   Number of Samples: {num_samples}")
        
        # Training duration (NEW - more detailed)
        training_duration = metrics.get('training_duration', metrics.get('training_time', 'N/A'))
        if isinstance(training_duration, (int, float)):
            self.logger.info(f"   Training Time: {training_duration:.3f}s")
        else:
            self.logger.info(f"   Training Time: {training_duration}")
        
        # Epochs completed (NEW)
        epochs_completed = metrics.get('epochs_completed', metrics.get('epochs', 'N/A'))
        self.logger.info(f"   Epochs Completed: {epochs_completed}")
        
        # Learning rate used (NEW)
        learning_rate = metrics.get('learning_rate', 'N/A')
        if isinstance(learning_rate, (int, float)):
            self.logger.info(f"   Learning Rate: {learning_rate:.6f}")
        else:
            self.logger.info(f"   Learning Rate: {learning_rate}")
        
        # Batch size used (NEW)
        batch_size = metrics.get('batch_size', 'N/A')
        self.logger.info(f"   Batch Size: {batch_size}")
        
        # Early stopping status (NEW)
        early_stopped = metrics.get('early_stopped', metrics.get('early_stopping', False))
        if early_stopped:
            self.logger.info(f"   Early Stopping: Yes")
            early_stopping_reason = metrics.get('early_stopping_reason', 'No improvement')
            self.logger.info(f"   Early Stopping Reason: {early_stopping_reason}")
            early_stopping_epoch = metrics.get('early_stopping_epoch', 'N/A')
            self.logger.info(f"   Stopped at Epoch: {early_stopping_epoch}")
        else:
            self.logger.info(f"   Early Stopping: No")
        
        # Additional metrics if available
        if 'loss_history' in metrics:
            loss_history = metrics['loss_history']
            if isinstance(loss_history, list) and len(loss_history) > 1:
                initial_loss = loss_history[0] if len(loss_history) > 0 else 0.0
                final_loss = loss_history[-1] if len(loss_history) > 0 else 0.0
                loss_improvement = initial_loss - final_loss
                self.logger.info(f"   Loss Improvement: {loss_improvement:.6f}")
        
        # Model convergence info
        converged = metrics.get('converged', 'N/A')
        if converged != 'N/A':
            self.logger.info(f"   Model Converged: {converged}")
            
        # Memory usage if available
        memory_usage = metrics.get('memory_usage_mb', 'N/A')
        if isinstance(memory_usage, (int, float)):
            self.logger.info(f"   Memory Usage: {memory_usage:.2f} MB")
        elif memory_usage != 'N/A':
            self.logger.info(f"   Memory Usage: {memory_usage}")
            
        # Local model update norm (for debugging)
        update_norm = metrics.get('update_norm', 'N/A')
        if isinstance(update_norm, (int, float)):
            self.logger.info(f"   Model Update Norm: {update_norm:.6f}")
        elif update_norm != 'N/A':
            self.logger.info(f"   Model Update Norm: {update_norm}")
        
        if 'early_stopping' in metrics and metrics['early_stopping']:
            self.log_early_stopping_event('client', client_id, round_num, metrics.get('early_stopping_reason', 'Unknown'))
    
    def log_global_aggregation(self, round_num: int, global_metrics: Dict[str, Any]):
        """Log global model aggregation results."""
        global_round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': global_metrics.copy()
        }
        self.global_metrics.append(global_round_data)
        
        self.logger.info(f"≡ƒîÉ Global Model Aggregation - Round {round_num + 1}:")
        
        # Safe formatting for numeric values
        global_accuracy = global_metrics.get('accuracy', 'N/A')
        if isinstance(global_accuracy, (int, float)):
            self.logger.info(f"   Global Accuracy: {global_accuracy:.6f}")
        else:
            self.logger.info(f"   Global Accuracy: {global_accuracy}")
            
        global_loss = global_metrics.get('loss', 'N/A')
        if isinstance(global_loss, (int, float)):
            self.logger.info(f"   Global Loss: {global_loss:.6f}")
        else:
            self.logger.info(f"   Global Loss: {global_loss}")
            
        num_participants = global_metrics.get('num_participants', 'N/A')
        self.logger.info(f"   Participating Clients: {num_participants}")
        
        aggregation_time = global_metrics.get('aggregation_time', 'N/A')
        if isinstance(aggregation_time, (int, float)):
            self.logger.info(f"   Aggregation Time: {aggregation_time:.3f}s")
        else:
            self.logger.info(f"   Aggregation Time: {aggregation_time}")
        
        if 'early_stopping' in global_metrics and global_metrics['early_stopping']:
            self.log_early_stopping_event('global', 'server', round_num, global_metrics.get('early_stopping_reason', 'Unknown'))
    
    def log_round_summary(self, round_num: int, client_results: Dict[str, Dict], global_results: Dict[str, Any]):
        """Log complete round summary and display console output."""
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'client_results': client_results,
            'global_results': global_results,
            'duration': global_results.get('round_duration', 0.0)
        }
        self.round_metrics.append(round_data)
        
        # Console output (enhanced with key metrics)
        print("   Client Results:")
        for client_id, metrics in client_results.items():
            val_accuracy = metrics.get('val_accuracy', 0.0)
            train_accuracy = metrics.get('training_accuracy', metrics.get('train_accuracy', 0.0))
            train_loss = metrics.get('training_loss', 0.0)
            val_loss = metrics.get('val_loss', 0.0)
            num_samples = metrics.get('num_samples', 0)
            training_time = metrics.get('training_duration', metrics.get('training_time', 0.0))
            epochs = metrics.get('epochs_completed', metrics.get('epochs', 'N/A'))
            early_stopped = metrics.get('early_stopped', metrics.get('early_stopping', False))
            
            # Format the output with additional key metrics
            early_stop_indicator = " [ES]" if early_stopped else ""
            time_str = f"{training_time:.1f}s" if isinstance(training_time, (int, float)) else str(training_time)
            
            print(f"     {client_id}: Train_Acc={train_accuracy:.3f}, Val_Acc={val_accuracy:.3f}, " +
                  f"Train_Loss={train_loss:.3f}, Val_Loss={val_loss:.3f}, " +
                  f"Time={time_str}, Epochs={epochs}, Samples={num_samples}{early_stop_indicator}")
        
        print(f"   ≡ƒÄ» Global Model: Acc={global_results.get('accuracy', 0.0):.3f}, Loss={global_results.get('loss', 0.0):.3f}")
        
        # Detailed file logging
        self.logger.info(f"\n≡ƒôè ROUND {round_num + 1} SUMMARY:")
        
        round_duration = global_results.get('round_duration', 0.0)
        if isinstance(round_duration, (int, float)):
            self.logger.info(f"   Round Duration: {round_duration:.3f}s")
        else:
            self.logger.info(f"   Round Duration: {round_duration}")
            
        self.logger.info(f"   Total Clients: {len(client_results)}")
        
        accuracy = global_results.get('accuracy', 0.0)
        loss = global_results.get('loss', 0.0)
        if isinstance(accuracy, (int, float)) and isinstance(loss, (int, float)):
            self.logger.info(f"   Global Performance: Acc={accuracy:.6f}, Loss={loss:.6f}")
        else:
            self.logger.info(f"   Global Performance: Acc={accuracy}, Loss={loss}")
        
        # Save round data immediately
        self._save_round_data(round_data)
    
    def log_early_stopping_event(self, level: str, entity_id: str, round_num: int, reason: str):
        """Log early stopping events."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'level': level,  # 'client' or 'global'
            'entity_id': entity_id,
            'round': round_num,
            'reason': reason
        }
        self.early_stopping_events.append(event)
        
        self.logger.warning(f"ΓÅ╣∩╕Å EARLY STOPPING - {level.upper()}")
        self.logger.warning(f"   Entity: {entity_id}")
        self.logger.warning(f"   Round: {round_num + 1}")
        self.logger.warning(f"   Reason: {reason}")
        
        print(f"ΓÅ╣∩╕Å Early stopping triggered for {entity_id} at round {round_num + 1}: {reason}")
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Log errors with context."""
        error_event = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        self.error_log.append(error_event)
        
        self.logger.error(f"Γ¥î ERROR - {error_type}")
        self.logger.error(f"   Message: {error_message}")
        if context:
            self.logger.error(f"   Context: {json.dumps(context, indent=2, default=str)}")
    
    def finalize_experiment(self, final_results: Dict[str, Any], status: str = 'completed'):
        """Finalize experiment and save all results."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.experiment_metadata.update({
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'duration_formatted': str(end_time - self.start_time),
            'status': status,
            'final_results': final_results,
            'total_rounds_completed': len(self.round_metrics),
            'early_stopping_events': len(self.early_stopping_events),
            'errors_count': len(self.error_log)
        })
        
        # Save all results
        self._save_experiment_metadata()
        self._save_client_metrics()
        self._save_global_metrics()
        self._save_round_metrics()
        self._save_early_stopping_events()
        self._save_error_log()
        self._generate_summary_report()
        self._generate_csv_exports()
        
        self.logger.info(f"\n≡ƒÄè EXPERIMENT COMPLETED")
        self.logger.info(f"   Status: {status}")
        self.logger.info(f"   Duration: {self.experiment_metadata['duration_formatted']}")
        self.logger.info(f"   Rounds Completed: {self.experiment_metadata['total_rounds_completed']}")
        self.logger.info(f"   Early Stopping Events: {self.experiment_metadata['early_stopping_events']}")
        self.logger.info(f"   Errors: {self.experiment_metadata['errors_count']}")
        self.logger.info(f"   Results Directory: {self.experiment_dir}")
        
        print(f"\nΓ£à Experiment {self.experiment_id} completed in {self.experiment_metadata['duration_formatted']}")
        print(f"≡ƒôü Results saved to: {self.experiment_dir}")
    
    def _save_round_data(self, round_data: Dict[str, Any]):
        """Save individual round data immediately."""
        round_file = self.raw_data_dir / f"round_{round_data['round']:03d}.json"
        with open(round_file, 'w') as f:
            json.dump(round_data, f, indent=2, default=str)
    
    def _save_experiment_metadata(self):
        """Save experiment metadata."""
        metadata_file = self.experiment_dir / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, default=str)
    
    def _save_client_metrics(self):
        """Save detailed client metrics."""
        client_file = self.metrics_dir / "client_metrics.json"
        with open(client_file, 'w') as f:
            json.dump(self.client_metrics, f, indent=2, default=str)
    
    def _save_global_metrics(self):
        """Save global metrics."""
        global_file = self.metrics_dir / "global_metrics.json"
        with open(global_file, 'w') as f:
            json.dump(self.global_metrics, f, indent=2, default=str)
    
    def _save_round_metrics(self):
        """Save round-by-round metrics."""
        round_file = self.metrics_dir / "round_metrics.json"
        with open(round_file, 'w') as f:
            json.dump(self.round_metrics, f, indent=2, default=str)
    
    def _save_early_stopping_events(self):
        """Save early stopping events."""
        if self.early_stopping_events:
            es_file = self.metrics_dir / "early_stopping_events.json"
            with open(es_file, 'w') as f:
                json.dump(self.early_stopping_events, f, indent=2, default=str)
    
    def _save_error_log(self):
        """Save error log."""
        if self.error_log:
            error_file = self.logs_dir / "errors.json"
            with open(error_file, 'w') as f:
                json.dump(self.error_log, f, indent=2, default=str)
    
    def _generate_summary_report(self):
        """Generate human-readable summary report."""
        summary_file = self.experiment_dir / "experiment_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Experiment Summary: {self.experiment_id}\n\n")
            f.write(f"**Status:** {self.experiment_metadata['status']}\n")
            f.write(f"**Duration:** {self.experiment_metadata.get('duration_formatted', 'N/A')}\n")
            f.write(f"**Rounds Completed:** {self.experiment_metadata.get('total_rounds_completed', 0)}\n\n")
            
            f.write("## Configuration\n")
            f.write(f"- Algorithm: {self.experiment_metadata.get('algorithm', 'N/A')}\n")
            f.write(f"- Model: {self.experiment_metadata.get('model_type', 'N/A')}\n")
            f.write(f"- Clients: {self.experiment_metadata.get('num_clients', 'N/A')}\n")
            f.write(f"- Planned Rounds: {self.experiment_metadata.get('num_rounds', 'N/A')}\n\n")
            
            if self.round_metrics:
                f.write("## Performance Summary\n")
                final_round = self.round_metrics[-1]
                f.write(f"- Final Global Accuracy: {final_round['global_results'].get('accuracy', 'N/A'):.4f}\n")
                f.write(f"- Final Global Loss: {final_round['global_results'].get('loss', 'N/A'):.4f}\n\n")
            
            if self.early_stopping_events:
                f.write("## Early Stopping Events\n")
                for event in self.early_stopping_events:
                    f.write(f"- Round {event['round'] + 1}: {event['entity_id']} - {event['reason']}\n")
                f.write("\n")
            
            if self.error_log:
                f.write("## Errors\n")
                for error in self.error_log:
                    f.write(f"- {error['type']}: {error['message']}\n")
    
    def _generate_csv_exports(self):
        """Generate CSV files for data analysis."""
        try:
            # Round-by-round summary
            if self.round_metrics:
                round_data = []
                for round_info in self.round_metrics:
                    row = {
                        'round': round_info['round'],
                        'global_accuracy': round_info['global_results'].get('accuracy', 0.0),
                        'global_loss': round_info['global_results'].get('loss', 0.0),
                        'num_participants': round_info['global_results'].get('num_participants', 0),
                        'round_duration': round_info['global_results'].get('round_duration', 0.0)
                    }
                    
                    # Add client averages
                    client_accs = [cr.get('val_accuracy', 0.0) for cr in round_info['client_results'].values()]
                    client_losses = [cr.get('training_loss', 0.0) for cr in round_info['client_results'].values()]
                    
                    row['avg_client_accuracy'] = np.mean(client_accs) if client_accs else 0.0
                    row['avg_client_loss'] = np.mean(client_losses) if client_losses else 0.0
                    
                    round_data.append(row)
                
                df_rounds = pd.DataFrame(round_data)
                df_rounds.to_csv(self.metrics_dir / "round_summary.csv", index=False)
            
            # Client performance details
            if self.client_metrics:
                client_data = []
                for client_id, rounds in self.client_metrics.items():
                    for round_info in rounds:
                        row = {
                            'client_id': client_id,
                            'round': round_info['round'],
                            'timestamp': round_info['timestamp']
                        }
                        row.update(round_info['metrics'])
                        client_data.append(row)
                
                df_clients = pd.DataFrame(client_data)
                df_clients.to_csv(self.metrics_dir / "client_details.csv", index=False)
                
        except Exception as e:
            self.logger.error(f"Failed to generate CSV exports: {e}")
