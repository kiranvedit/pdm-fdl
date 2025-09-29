"""
Experiment Orchestrator
======================
Main orchestrator for running federated learning experiments using PySyft.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.interfaces import OrchestratorInterface, ModelInterface
from core.enums import FederatedAlgorithm, ModelType, ExperimentStatus
from core.exceptions import ExperimentError, CommunicationError

# Import ModelFactory from shared models - fix the import path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, '..', '..', '..', 'shared')
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

try:
    from models.model_factory import ModelFactory
    # ModelManager doesn't exist, we'll create it or remove the reference
except ImportError:
    # Create a simple ModelManager class if needed
    class ModelManager:
        pass

from federation.algorithms.fedavg import FedAvgAlgorithm
from federation.algorithms.fedprox import FedProxAlgorithm
from federation.communication.syft_client import PySyftCommunicationManager
from monitoring.metrics_collector import MetricsCollector
from .experiment_config import ExperimentConfig
from .results_collector import ResultsCollector


class FederatedExperimentOrchestrator(OrchestratorInterface):
    """
    Main orchestrator for federated learning experiments.
    Coordinates models, algorithms, communication, and metrics collection.
    """
    
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Core components
        self.model_manager = ModelManager()
        self.communication_manager = PySyftCommunicationManager()
        self.metrics_collector = MetricsCollector()
        self.results_collector = ResultsCollector(str(self.base_results_dir))
        
        # Algorithm registry
        self.algorithm_registry = {
            FederatedAlgorithm.FEDAVG: FedAvgAlgorithm,
            FederatedAlgorithm.FEDPROX: FedProxAlgorithm,
            # Add other algorithms as implemented
        }
        
        # Experiment tracking
        self.active_experiments = {}
        self.experiment_history = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def setup_experiment(self, config: Dict[str, Any]) -> None:
        """
        Setup experiment with configuration.
        
        Args:
            config: Experiment configuration dictionary
        """
        try:
            # Parse and validate configuration
            experiment_config = ExperimentConfig.from_dict(config)
            experiment_id = experiment_config.experiment_id
            
            self.logger.info(f"Setting up experiment {experiment_id}")
            
            # Create model for the experiment
            model = self.model_manager.create_and_register_model(
                model_id=f"{experiment_id}_model",
                model_type=experiment_config.model_type,
                **experiment_config.model_params
            )
            
            # Initialize algorithm
            algorithm_class = self.algorithm_registry.get(experiment_config.algorithm)
            if not algorithm_class:
                raise ExperimentError(f"Unsupported algorithm: {experiment_config.algorithm}")
            
            algorithm = algorithm_class(**experiment_config.algorithm_params)
            
            # Connect to datasites
            connected_datasites = []
            for datasite_config in experiment_config.datasites:
                success = self.communication_manager.connect_to_datasite(
                    datasite_config['url'],
                    datasite_config['credentials']
                )
                if success:
                    connected_datasites.append(datasite_config['datasite_id'])
                else:
                    self.logger.warning(f"Failed to connect to {datasite_config['datasite_id']}")
            
            if not connected_datasites:
                raise CommunicationError("No datasites connected successfully")
            
            # Store experiment setup
            self.active_experiments[experiment_id] = {
                'config': experiment_config,
                'model': model,
                'algorithm': algorithm,
                'connected_datasites': connected_datasites,
                'status': ExperimentStatus.PENDING,
                'created_at': datetime.now(),
                'results': {}
            }
            
            self.logger.info(f"Experiment {experiment_id} setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup experiment: {e}")
            raise ExperimentError(f"Experiment setup failed: {str(e)}")
    
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Run a single federated learning experiment.
        
        Args:
            experiment_id: ID of the experiment to run
            
        Returns:
            Dict with experiment results
        """
        if experiment_id not in self.active_experiments:
            raise ExperimentError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        config = experiment['config']
        model = experiment['model']
        algorithm = experiment['algorithm']
        connected_datasites = experiment['connected_datasites']
        
        try:
            self.logger.info(f"Starting experiment {experiment_id}")
            experiment['status'] = ExperimentStatus.RUNNING
            
            # Initialize results tracking
            experiment_results = {
                'experiment_id': experiment_id,
                'algorithm': config.algorithm.value,
                'model_type': config.model_type.value,
                'max_rounds': config.max_rounds,
                'datasites': connected_datasites,
                'round_results': [],
                'final_metrics': {},
                'start_time': datetime.now(),
                'end_time': None
            }
            
            # Run federated learning rounds
            for round_num in range(config.max_rounds):
                self.logger.info(f"Round {round_num + 1}/{config.max_rounds}")
                
                round_results = self._run_federated_round(
                    round_num + 1,
                    model,
                    algorithm,
                    connected_datasites,
                    config
                )
                
                experiment_results['round_results'].append(round_results)
                
                # Check for early stopping
                if self._should_stop_early(experiment_results, config):
                    self.logger.info(f"Early stopping triggered at round {round_num + 1}")
                    break
                
                # Update metrics
                self.metrics_collector.collect_round_metrics(
                    round_num + 1, model, round_results['client_updates']
                )
            
            # Finalize experiment
            experiment_results['end_time'] = datetime.now()
            experiment_results['final_metrics'] = self.metrics_collector.get_final_metrics()
            
            # Store results
            experiment['results'] = experiment_results
            experiment['status'] = ExperimentStatus.COMPLETED
            
            # Save results to file
            self.results_collector.save_experiment_results(experiment_id, experiment_results)
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment['status'] = ExperimentStatus.FAILED
            raise ExperimentError(f"Experiment execution failed: {str(e)}")
    
    def _run_federated_round(self, round_num: int, model: ModelInterface,
                           algorithm, connected_datasites: List[str],
                           config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single federated learning round."""
        
        # Get current global model parameters
        global_params = model.get_parameters()
        
        # Configure client training for this round
        training_config = algorithm.configure_client_training(round_num)
        training_config.update(config.training_params)
        
        # Collect client updates
        client_updates = []
        round_start_time = time.time()
        
        for datasite_id in connected_datasites:
            try:
                # Send global model to datasite
                success = self.communication_manager.send_model(global_params, datasite_id)
                if not success:
                    self.logger.warning(f"Failed to send model to {datasite_id}")
                    continue
                
                # Request local training
                training_results = self.communication_manager.request_training(
                    training_config, datasite_id
                )
                
                if 'error' not in training_results:
                    client_updates.append({
                        'datasite_id': datasite_id,
                        **training_results
                    })
                else:
                    self.logger.warning(f"Training failed on {datasite_id}: {training_results['error']}")
                
            except Exception as e:
                self.logger.error(f"Error processing datasite {datasite_id}: {e}")
                continue
        
        if not client_updates:
            raise ExperimentError(f"No successful client updates in round {round_num}")
        
        # Aggregate client updates
        aggregated_params = algorithm.aggregate(client_updates)
        
        # Update global model
        model.set_parameters(aggregated_params)
        
        # Update algorithm state
        algorithm.update_algorithm_state(client_updates)
        
        round_time = time.time() - round_start_time
        
        return {
            'round_num': round_num,
            'num_participants': len(client_updates),
            'client_updates': client_updates,
            'aggregated_params': aggregated_params,
            'round_time': round_time,
            'algorithm_metrics': algorithm.get_algorithm_specific_metrics(client_updates)
        }
    
    def _should_stop_early(self, experiment_results: Dict[str, Any], 
                          config: ExperimentConfig) -> bool:
        """Check if early stopping criteria are met."""
        if not config.early_stopping_enabled:
            return False
        
        round_results = experiment_results['round_results']
        if len(round_results) < config.early_stopping_patience:
            return False
        
        # Simple convergence check based on loss improvement
        recent_losses = []
        for result in round_results[-config.early_stopping_patience:]:
            avg_loss = sum(update.get('loss', float('inf')) 
                          for update in result['client_updates']) / len(result['client_updates'])
            recent_losses.append(avg_loss)
        
        # Check if loss improvement is below threshold
        if len(recent_losses) >= 2:
            improvement = abs(recent_losses[-1] - recent_losses[-2])
            return improvement < config.early_stopping_threshold
        
        return False
    
    def collect_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Collect and aggregate experiment results.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dict with aggregated results
        """
        if experiment_id not in self.active_experiments:
            raise ExperimentError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        if 'results' not in experiment:
            raise ExperimentError(f"No results available for experiment {experiment_id}")
        
        return experiment['results']
    
    def run_experiment_batch(self, config_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a batch of experiments.
        
        Args:
            config_list: List of experiment configurations
            
        Returns:
            Dict with batch results
        """
        batch_results = {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_experiments': len(config_list),
            'completed': 0,
            'failed': 0,
            'experiment_results': {}
        }
        
        for i, config in enumerate(config_list):
            try:
                experiment_id = config.get('experiment_id', f"exp_{i}")
                
                self.setup_experiment(config)
                results = self.run_experiment(experiment_id)
                
                batch_results['experiment_results'][experiment_id] = results
                batch_results['completed'] += 1
                
            except Exception as e:
                self.logger.error(f"Experiment {i} failed: {e}")
                batch_results['failed'] += 1
                batch_results['experiment_results'][f"failed_exp_{i}"] = {'error': str(e)}
        
        # Save batch results
        self.results_collector.save_batch_results(batch_results['batch_id'], batch_results)
        
        return batch_results
    
    def get_experiment_status(self, experiment_id: str) -> ExperimentStatus:
        """Get the status of an experiment."""
        if experiment_id not in self.active_experiments:
            return ExperimentStatus.PENDING
        return self.active_experiments[experiment_id]['status']
    
    def list_active_experiments(self) -> List[str]:
        """List all active experiment IDs."""
        return list(self.active_experiments.keys())
    
    def cleanup_experiment(self, experiment_id: str) -> None:
        """Clean up resources for an experiment."""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            
            # Disconnect from datasites
            for datasite_id in experiment.get('connected_datasites', []):
                self.communication_manager.disconnect_from_datasite(datasite_id)
            
            # Remove model
            model_id = f"{experiment_id}_model"
            self.model_manager.remove_model(model_id)
            
            # Move to history
            self.experiment_history.append(self.active_experiments[experiment_id])
            del self.active_experiments[experiment_id]
            
            self.logger.info(f"Experiment {experiment_id} cleaned up")
    
    def cleanup_all_experiments(self) -> None:
        """Clean up all active experiments."""
        experiment_ids = list(self.active_experiments.keys())
        for experiment_id in experiment_ids:
            self.cleanup_experiment(experiment_id)
        
        # Clean up communication manager
        self.communication_manager.cleanup_all_connections()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_results_dir / 'orchestrator.log'),
                logging.StreamHandler()
            ]
        )
