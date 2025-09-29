#!/usr/bin/env python3
"""
Configurable Experiment Configuration System
Allows dynamic parameter setting and individual experiment execution
"""

import argparse
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ExperimentParams:
    """Configurable experiment parameters."""
    max_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    num_datasites: int = 3
    
    # Algorithm-specific parameters
    fedprox_mu: float = 0.01
    scaffold_lr_server: float = 1.0
    
    # Data distribution parameters
    alpha_dirichlet: float = 0.5  # For non-IID distribution
    
    # Communication parameters
    compression_rate: float = 0.1  # For compressed communication
    
    # Monitoring ports configuration
    heartbeat_port: int = 8888      # Port for heartbeat manager API
    dashboard_port: int = 8889      # Port for status dashboard web interface
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentParams':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'ExperimentParams':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class ExperimentState:
    """Manages experiment state and resumption."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.state_file = os.path.join(results_dir, "experiment_state.json")
        self.completed_experiments = set()
        self.current_round_by_experiment = {}
        self.load_state()
    
    def load_state(self):
        """Load experiment state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self.completed_experiments = set(data.get('completed_experiments', []))
                self.current_round_by_experiment = data.get('current_round_by_experiment', {})
                print(f"[RESUME] Loaded state: {len(self.completed_experiments)} completed experiments")
        except Exception as e:
            print(f"[WARNING] Failed to load experiment state: {e}")
    
    def save_state(self):
        """Save current experiment state."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state_data = {
                'completed_experiments': list(self.completed_experiments),
                'current_round_by_experiment': self.current_round_by_experiment,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save experiment state: {e}")
    
    def is_experiment_completed(self, experiment_name: str) -> bool:
        """Check if experiment is already completed."""
        return experiment_name in self.completed_experiments
    
    def mark_experiment_completed(self, experiment_name: str):
        """Mark experiment as completed."""
        self.completed_experiments.add(experiment_name)
        if experiment_name in self.current_round_by_experiment:
            del self.current_round_by_experiment[experiment_name]
        self.save_state()
    
    def get_current_round(self, experiment_name: str) -> int:
        """Get current round for experiment (0-based)."""
        return self.current_round_by_experiment.get(experiment_name, 0)
    
    def update_round(self, experiment_name: str, round_num: int):
        """Update current round for experiment."""
        self.current_round_by_experiment[experiment_name] = round_num
        self.save_state()
    
    def get_pending_experiments(self, all_experiments: List[str]) -> List[str]:
        """Get list of experiments that haven't been completed."""
        return [exp for exp in all_experiments if not self.is_experiment_completed(exp)]

class ConfigurableExperimentRunner:
    """Enhanced experiment runner with configurable parameters and resumption."""
    
    def __init__(self, params: ExperimentParams, results_dir: Optional[str] = None):
        self.params = params
        
        # Create results directory
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join("results", f"experiment_run_{timestamp}")
        
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        
        # Initialize state management
        self.state = ExperimentState(results_dir)
        
        # Save experiment configuration
        config_file = os.path.join(results_dir, "experiment_config.json")
        self.params.save_to_file(config_file)
        
        print(f"[CONFIG] Experiment configuration saved to: {config_file}")
        print(f"[RESULTS] Results directory: {results_dir}")
    
    def get_all_experiment_names(self) -> List[str]:
        """Generate all possible experiment combinations."""
        algorithms = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova']
        models = ['OptimizedCNN', 'OptimizedLSTM', 'OptimizedHybrid']
        distributions = ['uniform', 'non_iid', 'heterogeneous', 'clustered']
        communications = ['sync', 'async', 'compressed']
        
        experiments = []
        for alg in algorithms:
            for model in models:
                for dist in distributions:
                    for comm in communications:
                        experiments.append(f"{alg}_{model}_{dist}_{comm}")
        
        return experiments
    
    def run_individual_experiment(self, experiment_name: str, resume: bool = True) -> Dict[str, Any]:
        """Run a single experiment with resumption support."""
        print(f"\n[EXPERIMENT] Starting: {experiment_name}")
        
        # Check if already completed
        if resume and self.state.is_experiment_completed(experiment_name):
            print(f"[SKIP] Experiment {experiment_name} already completed")
            return {'status': 'already_completed', 'experiment_name': experiment_name}
        
        # Get starting round
        start_round = self.state.get_current_round(experiment_name) if resume else 0
        if start_round > 0:
            print(f"[RESUME] Resuming from round {start_round + 1}")
        
        try:
            # Parse experiment name
            parts = experiment_name.split('_')
            if len(parts) != 4:
                raise ValueError(f"Invalid experiment name format: {experiment_name}")
            
            algorithm_name, model_name, distribution_name, communication_style = parts
            
            # Create experiment-specific results directory
            exp_dir = os.path.join(self.results_dir, experiment_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Run experiment with resumption
            result = self._run_federated_experiment(
                algorithm_name, model_name, distribution_name, communication_style,
                exp_dir, start_round
            )
            
            # Mark as completed if successful
            if result.get('status') == 'completed':
                self.state.mark_experiment_completed(experiment_name)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Experiment {experiment_name} failed: {e}")
            return {
                'status': 'failed',
                'experiment_name': experiment_name,
                'error': str(e)
            }
    
    def _run_federated_experiment(self, algorithm_name: str, model_name: str, 
                                 distribution_name: str, communication_style: str,
                                 exp_dir: str, start_round: int) -> Dict[str, Any]:
        """Run federated learning experiment with round-by-round state saving."""
        experiment_name = f"{algorithm_name}_{model_name}_{distribution_name}_{communication_style}"
        
        print(f"[FEDERATED] Running {experiment_name} (rounds {start_round + 1}-{self.params.max_rounds})")
        
        # Save round-by-round metrics
        round_metrics = []
        
        for round_num in range(start_round, self.params.max_rounds):
            print(f"[ROUND] {round_num + 1}/{self.params.max_rounds}")
            
            # Simulate round execution (replace with actual federated learning)
            round_result = self._execute_round(round_num, algorithm_name, model_name)
            round_metrics.append(round_result)
            
            # Save round state immediately
            self._save_round_state(exp_dir, round_num + 1, round_result)
            
            # Update experiment state
            self.state.update_round(experiment_name, round_num + 1)
            
            print(f"[PROGRESS] Round {round_num + 1} completed - Acc: {round_result.get('accuracy', 0.0):.4f}")
        
        # Save final results
        final_result = {
            'status': 'completed',
            'experiment_name': experiment_name,
            'params': self.params.to_dict(),
            'round_metrics': round_metrics,
            'final_accuracy': round_metrics[-1].get('accuracy', 0.0) if round_metrics else 0.0,
            'total_rounds': len(round_metrics)
        }
        
        result_file = os.path.join(exp_dir, "final_result.json")
        with open(result_file, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        return final_result
    
    def _execute_round(self, round_num: int, algorithm_name: str, model_name: str) -> Dict[str, Any]:
        """
        Execute a single federated learning round - NO MORE FAKE VALUES
        This method should use REAL federated learning results, not simulated ones.
        """
        # ERROR: This method should NOT contain hardcoded values!
        # This is a placeholder that must be replaced with real federated learning execution
        print(f"ERROR: _execute_round called with FAKE values - this should not happen!")
        print(f"ERROR: Round {round_num}, Algorithm: {algorithm_name}, Model: {model_name}")
        
        # Return zeros instead of fake values - this indicates real execution is needed
        return {
            'round': round_num + 1,
            'algorithm': algorithm_name,
            'model': model_name,
            'accuracy': 0.0,  # MUST be replaced with real training results
            'loss': 0.0,      # MUST be replaced with real training results
            'timestamp': datetime.now().isoformat(),
            'error': 'FAKE_SIMULATION_DETECTED'
        }
    
    def _save_round_state(self, exp_dir: str, round_num: int, round_data: Dict[str, Any]):
        """Save round state to file."""
        rounds_dir = os.path.join(exp_dir, "rounds")
        os.makedirs(rounds_dir, exist_ok=True)
        
        round_file = os.path.join(rounds_dir, f"round_{round_num:03d}.json")
        with open(round_file, 'w') as f:
            json.dump(round_data, f, indent=2)

def create_default_config() -> ExperimentParams:
    """Create default experiment configuration."""
    return ExperimentParams(
        max_rounds=10,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        num_datasites=3
    )

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Configurable Federated Learning Experiments")
    parser.add_argument('--config', type=str, help='Path to experiment config JSON file')
    parser.add_argument('--experiment', type=str, help='Run specific experiment (e.g., FedAvg_OptimizedCNN_uniform_sync)')
    parser.add_argument('--max-rounds', type=int, help='Maximum number of rounds')
    parser.add_argument('--local-epochs', type=int, help='Local training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--results-dir', type=str, help='Results directory path')
    parser.add_argument('--resume', action='store_true', help='Resume from previous state')
    parser.add_argument('--list-experiments', action='store_true', help='List all possible experiments')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        params = ExperimentParams.from_json_file(args.config)
    else:
        params = create_default_config()
    
    # Override with command line arguments
    if args.max_rounds:
        params.max_rounds = args.max_rounds
    if args.local_epochs:
        params.local_epochs = args.local_epochs
    if args.batch_size:
        params.batch_size = args.batch_size
    if args.learning_rate:
        params.learning_rate = args.learning_rate
    
    # Create runner
    runner = ConfigurableExperimentRunner(params, args.results_dir)
    
    if args.list_experiments:
        experiments = runner.get_all_experiment_names()
        print("All possible experiments:")
        for i, exp in enumerate(experiments, 1):
            print(f"{i:2d}. {exp}")
        return
    
    if args.experiment:
        # Run specific experiment
        result = runner.run_individual_experiment(args.experiment, resume=args.resume)
        print(f"Experiment result: {result}")
    else:
        print("Use --experiment to run a specific experiment or --list-experiments to see all options")

if __name__ == "__main__":
    main()
