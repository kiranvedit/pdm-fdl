"""
Results Storage and Management Module for Federated Learning Experiments

This module provides comprehensive results storage, loading, and management
for federated learning experiments with detailed round-by-round metrics.

Features:
- Automatic results directory creation and management
- Round-by-round metrics storage for each experiment
- Comprehensive experiment metadata tracking
- Easy loading and aggregation of stored results
- Backup and recovery capabilities
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


class ExperimentResultsManager:
    """
    Manages experiment results storage and retrieval with detailed round metrics.
    """
    
    def __init__(self, base_results_dir="results"):
        """
        Initialize the results manager.
        
        Args:
            base_results_dir: Base directory for storing results
        """
        self.base_results_dir = base_results_dir
        self.ensure_results_directory()
    
    def ensure_results_directory(self):
        """Create results directory structure if it doesn't exist."""
        os.makedirs(self.base_results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_results_dir, "individual_runs"), exist_ok=True)
        os.makedirs(os.path.join(self.base_results_dir, "aggregated"), exist_ok=True)
        os.makedirs(os.path.join(self.base_results_dir, "metadata"), exist_ok=True)
    
    def create_experiment_session(self, session_name=None):
        """
        Create a new experiment session directory.
        
        Args:
            session_name: Optional session name, defaults to timestamp
            
        Returns:
            str: Session directory path
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_dir = os.path.join(self.base_results_dir, "individual_runs", session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Create session metadata
        session_metadata = {
            "session_name": session_name,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        with open(os.path.join(session_dir, "session_metadata.json"), "w") as f:
            json.dump(session_metadata, f, indent=2)
        
        return session_dir
    
    def save_experiment_result(self, session_dir, run_number, experiment_key, result):
        """
        Save a single experiment result with full round metrics.
        
        Args:
            session_dir: Session directory path
            run_number: Run number (1, 2, 3, ...)
            experiment_key: Experiment identifier (e.g., "cnn_fedavg_iid_standard")
            result: Complete experiment result dictionary
        """
        run_dir = os.path.join(session_dir, f"run_{run_number}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save complete result as pickle for full data
        result_file = os.path.join(run_dir, f"{experiment_key}.pkl")
        with open(result_file, "wb") as f:
            pickle.dump(result, f)
        
        # Extract and save summary metrics as JSON for easy reading
        summary = self.extract_experiment_summary(result, experiment_key)
        summary_file = os.path.join(run_dir, f"{experiment_key}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save round-by-round metrics as CSV for analysis
        if 'round_metrics' in result and result['round_metrics']:
            rounds_df = pd.DataFrame(result['round_metrics'])
            rounds_df['experiment_key'] = experiment_key
            rounds_df['run_number'] = run_number
            rounds_file = os.path.join(run_dir, f"{experiment_key}_rounds.csv")
            rounds_df.to_csv(rounds_file, index=False)
    
    def extract_experiment_summary(self, result, experiment_key):
        """
        Extract key metrics summary from experiment result.
        
        Args:
            result: Complete experiment result
            experiment_key: Experiment identifier
            
        Returns:
            dict: Summary metrics
        """
        # Parse experiment key
        parts = experiment_key.split('_')
        model = parts[0] if len(parts) > 0 else "unknown"
        algorithm = parts[1] if len(parts) > 1 else "unknown"
        distribution = parts[2] if len(parts) > 2 else "unknown"
        server_type = parts[3] if len(parts) > 3 else "unknown"
        
        # Extract final metrics
        final_metrics = result.get('final_metrics', {})
        experiment_summary = result.get('experiment_summary', {})
        
        # Get final performance metrics
        final_accuracy = (final_metrics.get('current_accuracy') or 
                         final_metrics.get('best_accuracy') or 0.0)
        final_f1 = (final_metrics.get('current_f1_score') or 
                   final_metrics.get('best_f1_score') or 0.0)
        final_precision = (final_metrics.get('current_precision') or 
                          final_metrics.get('best_precision') or 0.0)
        final_recall = (final_metrics.get('current_recall') or 
                       final_metrics.get('best_recall') or 0.0)
        final_auc = (final_metrics.get('current_auc') or 
                    final_metrics.get('best_auc') or 0.0)
        
        # Get timing metrics
        training_time = experiment_summary.get('total_training_time', 0.0)
        avg_round_time = experiment_summary.get('avg_round_time', 0.0)
        
        # Calculate inference time from client metrics if available
        inference_time = 0.0
        if 'client_metrics' in result and result['client_metrics']:
            total_inference = sum(cm.get('inference_time', 0.0) 
                                for cm in result['client_metrics'])
            inference_time = total_inference / len(result['client_metrics'])
        elif final_metrics.get('avg_client_inference_time'):
            inference_time = final_metrics.get('avg_client_inference_time', 0.0)
        
        # Fallback to round metrics if final metrics are missing
        if final_accuracy == 0.0 and 'round_metrics' in result and result['round_metrics']:
            last_round = result['round_metrics'][-1]
            final_accuracy = last_round.get('accuracy', 0.0)
            final_f1 = last_round.get('f1_score', 0.0)
            final_precision = last_round.get('precision', 0.0)
            final_recall = last_round.get('recall', 0.0)
            final_auc = last_round.get('auc', 0.0)
        
        return {
            "experiment_key": experiment_key,
            "model": model,
            "algorithm": algorithm,
            "distribution": distribution,
            "server_type": server_type,
            "final_accuracy": float(final_accuracy),
            "final_f1": float(final_f1),
            "final_precision": float(final_precision),
            "final_recall": float(final_recall),
            "final_auc": float(final_auc),
            "training_time": float(training_time),
            "inference_time": float(inference_time),
            "avg_round_time": float(avg_round_time),
            "total_rounds": experiment_summary.get('total_rounds_completed', 0),
            "early_stopping": experiment_summary.get('early_stopping_triggered', False),
            "status": "success" if final_accuracy > 0 else "failed",
            "timestamp": datetime.now().isoformat()
        }
    
    def load_session_results(self, session_name):
        """
        Load all results from a specific session.
        
        Args:
            session_name: Name of the session to load
            
        Returns:
            dict: Loaded results in the same format as experiment output
        """
        session_dir = os.path.join(self.base_results_dir, "individual_runs", session_name)
        
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        all_results = {}
        
        # Find all run directories
        for item in os.listdir(session_dir):
            if item.startswith("run_") and os.path.isdir(os.path.join(session_dir, item)):
                run_number = item
                run_dir = os.path.join(session_dir, run_number)
                run_results = {}
                
                # Load all experiment results from this run
                for file in os.listdir(run_dir):
                    if file.endswith(".pkl") and not file.endswith("_summary.json"):
                        experiment_key = file.replace(".pkl", "")
                        result_file = os.path.join(run_dir, file)
                        
                        with open(result_file, "rb") as f:
                            result = pickle.load(f)
                            run_results[experiment_key] = result
                
                if run_results:
                    all_results[run_number] = run_results
        
        return all_results
    
    def load_session_summaries(self, session_name):
        """
        Load summary data from a session for quick analysis.
        
        Args:
            session_name: Name of the session to load
            
        Returns:
            pd.DataFrame: Summary data as DataFrame
        """
        session_dir = os.path.join(self.base_results_dir, "individual_runs", session_name)
        
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        all_summaries = []
        
        # Find all run directories
        for item in os.listdir(session_dir):
            if item.startswith("run_") and os.path.isdir(os.path.join(session_dir, item)):
                run_number = int(item.replace("run_", ""))
                run_dir = os.path.join(session_dir, item)
                
                # Load all summary files from this run
                for file in os.listdir(run_dir):
                    if file.endswith("_summary.json"):
                        summary_file = os.path.join(run_dir, file)
                        
                        with open(summary_file, "r") as f:
                            summary = json.load(f)
                            summary['run_number'] = run_number
                            all_summaries.append(summary)
        
        return pd.DataFrame(all_summaries)
    
    def list_available_sessions(self):
        """
        List all available experiment sessions.
        
        Returns:
            list: List of session names
        """
        sessions_dir = os.path.join(self.base_results_dir, "individual_runs")
        
        if not os.path.exists(sessions_dir):
            return []
        
        sessions = []
        for item in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, item)
            if os.path.isdir(session_path):
                # Check if it has session metadata
                metadata_file = os.path.join(session_path, "session_metadata.json")
                if os.path.exists(metadata_file):
                    sessions.append(item)
        
        return sorted(sessions)
    
    def get_session_info(self, session_name):
        """
        Get information about a specific session.
        
        Args:
            session_name: Name of the session
            
        Returns:
            dict: Session information
        """
        session_dir = os.path.join(self.base_results_dir, "individual_runs", session_name)
        metadata_file = os.path.join(session_dir, "session_metadata.json")
        
        if not os.path.exists(metadata_file):
            return None
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Count runs and experiments
        runs = 0
        total_experiments = 0
        
        for item in os.listdir(session_dir):
            if item.startswith("run_") and os.path.isdir(os.path.join(session_dir, item)):
                runs += 1
                run_dir = os.path.join(session_dir, item)
                experiments = sum(1 for f in os.listdir(run_dir) if f.endswith(".pkl"))
                total_experiments += experiments
        
        metadata.update({
            "total_runs": runs,
            "total_experiments": total_experiments,
            "avg_experiments_per_run": total_experiments / runs if runs > 0 else 0
        })
        
        return metadata


def create_results_storage_enhanced_experiment_framework():
    """
    Create an enhanced version of the experiment framework with results storage.
    
    Returns:
        Enhanced experiment functions with automatic results storage
    """
    
    def run_multiple_comprehensive_experiments_with_storage(
        num_runs=32, max_rounds=50, local_epochs=10, num_clients=5, 
        verbose=True, session_name=None, save_results=True):
        """
        Enhanced version of run_multiple_comprehensive_experiments with automatic storage.
        """
        # Import required modules within function
        import time
        import torch
        from sklearn.model_selection import train_test_split
        from data_utils import load_model_specific_data, create_data_distribution, create_federated_clients
        from federated_server import FederatedServer
        from secure_federated_server import SecureFederatedServer
        from experiment_framework import run_federated_experiment
        from experiment_logger import setup_experiment_logging
        
        # Import model classes
        import sys
        import os
        shared_path = os.path.abspath('../../shared')
        if shared_path not in sys.path:
            sys.path.append(shared_path)
        from models.step1a_optimized_models import OptimizedCNNModel, OptimizedLSTMModel, OptimizedHybridModel
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize results manager
        results_manager = ExperimentResultsManager() if save_results else None
        session_dir = None
        
        # Initialize experiment logger
        logger = setup_experiment_logging(log_dir="logs", log_level="INFO")
        experiment_session_name = session_name or f"experiment_{int(time.time())}"
        logger.start_session(experiment_session_name)
        
        if save_results:
            session_dir = results_manager.create_experiment_session(session_name)
            print(f"📁 Results will be saved to: {session_dir}")
        
        print("="*100)
        print("ENHANCED COMPREHENSIVE FEDERATED LEARNING EXPERIMENTS")
        print("="*100)
        print(f"Experiment Configuration:")
        print(f"  Number of runs: {num_runs}")
        print(f"  Max rounds per experiment: {max_rounds}")
        print(f"  Local epochs per round: {local_epochs}")
        print(f"  Number of clients: {num_clients}")
        print(f"  Results storage: {'✅ Enabled' if save_results else '❌ Disabled'}")
        print(f"  Total experiments per run: 48 (3 models × 4 algorithms × 2 distributions × 2 servers)")
        print(f"  Total experiments overall: {num_runs * 48}")
        print("="*100)
        
        # Model configurations
        models_config = {
            'cnn': {
                'class': OptimizedCNNModel,
                'params': {'input_dim': 10, 'num_classes': 2},
                'data_type': 'tabular'
            },
            'lstm': {
                'class': OptimizedLSTMModel,
                'params': {'input_dim': 10, 'num_classes': 2, 'sequence_length': 10},
                'data_type': 'sequences'
            },
            'hybrid': {
                'class': OptimizedHybridModel,
                'params': {'input_dim': 10, 'num_classes': 2, 'sequence_length': 10},
                'data_type': 'tabular'
            }
        }
        
        algorithms_to_test = ['fedavg', 'feddyn', 'fedprox', 'fednova']
        distributions_to_test = ['iid', 'non_iid']
        server_types_to_test = ['standard', 'secure']
        
        # Storage for all runs
        all_runs_results = {}
        run_summaries = []
        
        total_start_time = time.time()
        
        for run_number in range(1, num_runs + 1):
            # Start logging for this run
            logger.start_run(run_number, num_runs)
            
            run_start_time = time.time()
            run_results = {}
            experiment_count = 0
            successful_experiments = 0
            failed_experiments = 0
            
            for model_name, model_config in models_config.items():
                for algorithm in algorithms_to_test:
                    for distribution_type in distributions_to_test:
                        for server_type in server_types_to_test:
                            experiment_count += 1
                            exp_key = f"{model_name}_{algorithm}_{distribution_type}_{server_type}"
                            
                            # Start logging for this experiment
                            experiment_config = {
                                'model': model_name,
                                'algorithm': algorithm,
                                'distribution': distribution_type,
                                'server_type': server_type,
                                'max_rounds': max_rounds,
                                'local_epochs': local_epochs,
                                'num_clients': num_clients
                            }
                            logger.start_experiment(exp_key, experiment_config)
                            
                            if verbose:
                                print(f"  [{experiment_count}/48] Running: {exp_key}")
                            
                            try:
                                # Load appropriate data
                                if model_config['data_type'] == 'tabular':
                                    X_train, X_test_data, y_train, y_test_data, data_info = load_model_specific_data('tabular')
                                    if model_name == 'hybrid':
                                        X_train = X_train.unsqueeze(1)
                                        X_test_data = X_test_data.unsqueeze(1)
                                else:
                                    X_train, X_test_data, y_train, y_test_data, data_info = load_model_specific_data('sequences')
                                
                                # Split data
                                X_fed, X_test_exp, y_fed, y_test_exp = train_test_split(
                                    X_train, y_train, test_size=0.2, random_state=42 + run_number, stratify=y_train
                                )
                                
                                # Create data distribution
                                client_data_map = create_data_distribution(
                                    X_fed.cpu().numpy(), y_fed.cpu().numpy(), 
                                    num_clients=num_clients, 
                                    distribution_type=distribution_type
                                )
                                
                                # Create clients
                                clients_exp = create_federated_clients(
                                    client_data_map=client_data_map,
                                    model_class=model_config['class'],
                                    model_params=model_config['params'],
                                    device=str(device),
                                    local_epochs=local_epochs
                                )
                                
                                # Create server
                                if server_type == 'secure':
                                    server_exp = SecureFederatedServer(
                                        model_class=model_config['class'],
                                        model_params=model_config['params'],
                                        aggregation_method=algorithm,
                                        device=str(device),
                                        enable_dp=True,
                                        enable_bft=True
                                    )
                                else:
                                    server_exp = FederatedServer(
                                        model_class=model_config['class'],
                                        model_params=model_config['params'],
                                        aggregation_method=algorithm,
                                        device=str(device)
                                    )
                                
                                # Run experiment
                                result = run_federated_experiment(
                                    server_exp,
                                    clients_exp,
                                    X_test_exp,
                                    y_test_exp,
                                    algorithm=algorithm,
                                    max_rounds=max_rounds
                                )
                                
                                # Save complete result with all round metrics
                                if save_results and session_dir:
                                    results_manager.save_experiment_result(
                                        session_dir, run_number, exp_key, result
                                    )
                                
                                # Extract summary metrics for in-memory storage
                                summary = results_manager.extract_experiment_summary(result, exp_key) if results_manager else {}
                                
                                run_results[exp_key] = {
                                    'model': model_name,
                                    'algorithm': algorithm,
                                    'distribution': distribution_type,
                                    'server_type': server_type,
                                    'final_accuracy': summary.get('final_accuracy', 0.0),
                                    'final_f1': summary.get('final_f1', 0.0),
                                    'precision': summary.get('final_precision', 0.0),
                                    'recall': summary.get('final_recall', 0.0),
                                    'auc': summary.get('final_auc', 0.0),
                                    'training_time': summary.get('training_time', 0.0),
                                    'inference_time': summary.get('inference_time', 0.0),
                                    'status': 'success'
                                }
                                
                                # Log experiment completion
                                logger.log_experiment_complete(exp_key, result)
                                successful_experiments += 1
                                
                            except Exception as e:
                                print(f"    ❌ Failed: {exp_key} - {str(e)}")
                                
                                # Log experiment error
                                logger.log_experiment_error(exp_key, str(e))
                                
                                run_results[exp_key] = {
                                    'model': model_name,
                                    'algorithm': algorithm,
                                    'distribution': distribution_type,
                                    'server_type': server_type,
                                    'error': str(e),
                                    'status': 'failed'
                                }
                                failed_experiments += 1
            
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            run_summary = {
                'run_number': run_number,
                'total_experiments': experiment_count,
                'successful_experiments': successful_experiments,
                'failed_experiments': failed_experiments,
                'success_rate': (successful_experiments / experiment_count) * 100,
                'run_duration': run_duration
            }
            
            run_summaries.append(run_summary)
            all_runs_results[f'run_{run_number}'] = run_results
            
            # Log run summary
            logger.display_run_summary(run_number, run_results)
            
            print(f"\n📊 Run {run_number} Summary:")
            print(f"   ✅ Successful: {successful_experiments}/48 ({(successful_experiments/48)*100:.1f}%)")
            print(f"   ❌ Failed: {failed_experiments}/48 ({(failed_experiments/48)*100:.1f}%)")
            print(f"   ⏱️  Duration: {run_duration/60:.1f} minutes")
            if save_results:
                print(f"   💾 Results saved to disk")
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Save session completion status
        if save_results and session_dir:
            session_metadata_file = os.path.join(session_dir, "session_metadata.json")
            with open(session_metadata_file, "r") as f:
                metadata = json.load(f)
            
            metadata.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "total_duration": total_duration,
                "num_runs": num_runs,
                "success_rate": sum(rs['success_rate'] for rs in run_summaries) / len(run_summaries)
            })
            
            with open(session_metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE EXPERIMENTS COMPLETED")
        print(f"{'='*100}")
        print(f"Total execution time: {total_duration/60:.1f} minutes")
        print(f"Average time per run: {(total_duration/num_runs)/60:.1f} minutes")
        if save_results:
            print(f"📁 All results saved to: {session_dir}")
            print(f"💾 Use ExperimentResultsManager to load and analyze results")
        
        # Close logging session
        logger.close_session()
        
        return {
            'all_runs_results': all_runs_results,
            'run_summaries': run_summaries,
            'session_dir': session_dir,
            'results_manager': results_manager
        }
    
    return run_multiple_comprehensive_experiments_with_storage


