"""
Metrics Collection System
========================
Collects and tracks various metrics during federated learning experiments.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from pathlib import Path

from core.interfaces import MetricsCollectorInterface, ModelInterface
from core.exceptions import ExperimentError


class MetricsCollector(MetricsCollectorInterface):
    """
    Comprehensive metrics collection for federated learning experiments.
    Tracks performance, communication, and algorithm-specific metrics.
    """
    
    def __init__(self):
        self.round_metrics = []
        self.experiment_metrics = {}
        self.communication_metrics = {}
        self.performance_history = []
        
        # Metric aggregators
        self.total_communication_time = 0.0
        self.total_training_time = 0.0
        self.total_aggregation_time = 0.0
    
    def collect_round_metrics(self, round_num: int, 
                            global_model: ModelInterface,
                            client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect comprehensive metrics for a single round.
        
        Args:
            round_num: Current round number
            global_model: Global model after aggregation
            client_updates: List of client update information
            
        Returns:
            Dict with round metrics
        """
        round_start_time = time.time()
        
        try:
            # Basic round information
            round_metrics = {
                'round_num': round_num,
                'timestamp': time.time(),
                'num_participants': len(client_updates),
                'total_samples': sum(update.get('num_samples', 0) for update in client_updates)
            }
            
            # Performance metrics from clients
            performance_metrics = self._collect_performance_metrics(client_updates)
            round_metrics.update(performance_metrics)
            
            # Communication metrics
            comm_metrics = self._collect_communication_metrics(client_updates)
            round_metrics.update(comm_metrics)
            
            # Training metrics
            training_metrics = self._collect_training_metrics(client_updates)
            round_metrics.update(training_metrics)
            
            # Model metrics (if global test data available)
            model_metrics = self._collect_model_metrics(global_model, round_num)
            round_metrics.update(model_metrics)
            
            # Convergence metrics
            convergence_metrics = self._collect_convergence_metrics(client_updates, round_num)
            round_metrics.update(convergence_metrics)
            
            # Store round metrics
            self.round_metrics.append(round_metrics)
            
            # Update aggregated metrics
            self._update_aggregated_metrics(round_metrics)
            
            collection_time = time.time() - round_start_time
            round_metrics['metrics_collection_time'] = collection_time
            
            return round_metrics
            
        except Exception as e:
            error_metrics = {
                'round_num': round_num,
                'error': str(e),
                'timestamp': time.time()
            }
            self.round_metrics.append(error_metrics)
            return error_metrics
    
    def _collect_performance_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect performance metrics from client updates."""
        if not client_updates:
            return {}
        
        # Extract performance metrics
        accuracies = [update.get('accuracy', 0) for update in client_updates]
        losses = [update.get('loss', 0) for update in client_updates]
        precisions = [update.get('precision', 0) for update in client_updates]
        recalls = [update.get('recall', 0) for update in client_updates]
        f1_scores = [update.get('f1_score', 0) for update in client_updates]
        
        # Calculate weighted averages (by number of samples)
        total_samples = sum(update.get('num_samples', 1) for update in client_updates)
        weights = [update.get('num_samples', 1) / total_samples for update in client_updates]
        
        return {
            'avg_accuracy': self._weighted_average(accuracies, weights),
            'avg_loss': self._weighted_average(losses, weights),
            'avg_precision': self._weighted_average(precisions, weights),
            'avg_recall': self._weighted_average(recalls, weights),
            'avg_f1_score': self._weighted_average(f1_scores, weights),
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'std_loss': np.std(losses) if losses else 0,
            'min_accuracy': min(accuracies) if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0,
            'client_performance_variance': np.var(accuracies) if accuracies else 0
        }
    
    def _collect_communication_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect communication-related metrics."""
        if not client_updates:
            return {}
        
        comm_times = [update.get('communication_time', 0) for update in client_updates]
        model_sizes = [update.get('model_size_mb', 0) for update in client_updates]
        
        total_comm_time = sum(comm_times)
        self.total_communication_time += total_comm_time
        
        return {
            'total_communication_time': total_comm_time,
            'avg_communication_time': np.mean(comm_times) if comm_times else 0,
            'max_communication_time': max(comm_times) if comm_times else 0,
            'total_data_transferred_mb': sum(model_sizes),
            'avg_model_size_mb': np.mean(model_sizes) if model_sizes else 0,
            'communication_efficiency': len(client_updates) / (total_comm_time + 1e-8)
        }
    
    def _collect_training_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect training-related metrics."""
        if not client_updates:
            return {}
        
        training_times = [update.get('training_time', 0) for update in client_updates]
        local_epochs = [update.get('local_epochs', 1) for update in client_updates]
        
        total_training_time = sum(training_times)
        self.total_training_time += total_training_time
        
        return {
            'total_training_time': total_training_time,
            'avg_training_time': np.mean(training_times) if training_times else 0,
            'max_training_time': max(training_times) if training_times else 0,
            'avg_local_epochs': np.mean(local_epochs) if local_epochs else 1,
            'training_efficiency': sum(local_epochs) / (total_training_time + 1e-8)
        }
    
    def _collect_model_metrics(self, global_model: ModelInterface, 
                             round_num: int) -> Dict[str, Any]:
        """Collect global model metrics."""
        try:
            # Model complexity metrics
            params = global_model.get_parameters()
            total_params = sum(p.numel() for p in params.values())
            model_size_mb = sum(p.element_size() * p.numel() for p in params.values()) / (1024 * 1024)
            
            # Parameter statistics
            param_norms = [torch.norm(p).item() for p in params.values()]
            
            return {
                'model_total_parameters': total_params,
                'model_size_mb': model_size_mb,
                'model_avg_param_norm': np.mean(param_norms) if param_norms else 0,
                'model_max_param_norm': max(param_norms) if param_norms else 0,
                'model_param_variance': np.var(param_norms) if param_norms else 0
            }
            
        except Exception as e:
            return {'model_metrics_error': str(e)}
    
    def _collect_convergence_metrics(self, client_updates: List[Dict[str, Any]], 
                                   round_num: int) -> Dict[str, Any]:
        """Collect convergence-related metrics."""
        if round_num < 2 or len(self.round_metrics) < 1:
            return {'convergence_rate': 0, 'is_converging': False}
        
        try:
            # Get current and previous round accuracies
            current_accuracy = self._weighted_average(
                [update.get('accuracy', 0) for update in client_updates],
                [update.get('num_samples', 1) for update in client_updates]
            )
            
            previous_accuracy = self.round_metrics[-1].get('avg_accuracy', 0)
            
            # Calculate convergence rate
            convergence_rate = current_accuracy - previous_accuracy
            
            # Check if converging (improving)
            is_converging = convergence_rate > 0
            
            # Calculate stability (variance in recent rounds)
            recent_accuracies = [rm.get('avg_accuracy', 0) for rm in self.round_metrics[-3:]]
            stability = 1.0 / (np.var(recent_accuracies) + 1e-8) if len(recent_accuracies) > 1 else 0
            
            return {
                'convergence_rate': convergence_rate,
                'is_converging': is_converging,
                'stability_metric': min(stability, 100.0),  # Cap for numerical stability
                'accuracy_improvement': convergence_rate > 0.001
            }
            
        except Exception as e:
            return {'convergence_metrics_error': str(e)}
    
    def collect_experiment_metrics(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect overall experiment metrics.
        
        Args:
            experiment_results: Complete experiment results
            
        Returns:
            Dict with experiment-level metrics
        """
        try:
            if not self.round_metrics:
                return {'error': 'No round metrics available'}
            
            # Overall performance trends
            accuracies = [rm.get('avg_accuracy', 0) for rm in self.round_metrics]
            losses = [rm.get('avg_loss', float('inf')) for rm in self.round_metrics]
            
            # Final performance
            final_accuracy = accuracies[-1] if accuracies else 0
            final_loss = losses[-1] if losses else float('inf')
            
            # Performance improvement
            initial_accuracy = accuracies[0] if accuracies else 0
            accuracy_improvement = final_accuracy - initial_accuracy
            
            # Convergence analysis
            convergence_round = self._find_convergence_round(accuracies)
            
            # Communication efficiency
            total_rounds = len(self.round_metrics)
            avg_participants = np.mean([rm.get('num_participants', 0) for rm in self.round_metrics])
            
            experiment_metrics = {
                'total_rounds': total_rounds,
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'accuracy_improvement': accuracy_improvement,
                'best_accuracy': max(accuracies) if accuracies else 0,
                'convergence_round': convergence_round,
                'avg_participants_per_round': avg_participants,
                'total_communication_time': self.total_communication_time,
                'total_training_time': self.total_training_time,
                'experiment_efficiency': final_accuracy / (self.total_training_time + 1e-8),
                'performance_stability': 1.0 / (np.var(accuracies[-5:]) + 1e-8) if len(accuracies) >= 5 else 0
            }
            
            self.experiment_metrics = experiment_metrics
            return experiment_metrics
            
        except Exception as e:
            return {'experiment_metrics_error': str(e)}
    
    def _find_convergence_round(self, accuracies: List[float], threshold: float = 0.001) -> Optional[int]:
        """Find the round where convergence occurred."""
        if len(accuracies) < 3:
            return None
        
        for i in range(2, len(accuracies)):
            # Check if improvement in last 3 rounds is below threshold
            recent_improvement = accuracies[i] - accuracies[i-2]
            if abs(recent_improvement) < threshold:
                return i + 1  # Return 1-indexed round number
        
        return None
    
    def _weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted average."""
        if not values or not weights:
            return 0.0
        
        try:
            return sum(v * w for v, w in zip(values, weights)) / sum(weights)
        except (ZeroDivisionError, TypeError):
            return np.mean(values) if values else 0.0
    
    def _update_aggregated_metrics(self, round_metrics: Dict[str, Any]) -> None:
        """Update running aggregated metrics."""
        # Update communication totals
        comm_time = round_metrics.get('total_communication_time', 0)
        training_time = round_metrics.get('total_training_time', 0)
        
        # Store for trend analysis
        self.performance_history.append({
            'round': round_metrics.get('round_num', 0),
            'accuracy': round_metrics.get('avg_accuracy', 0),
            'loss': round_metrics.get('avg_loss', 0),
            'participants': round_metrics.get('num_participants', 0)
        })
        
        # Keep only last 50 rounds for memory efficiency
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
    
    def get_final_metrics(self) -> Dict[str, Any]:
        """Get final aggregated metrics."""
        if not self.round_metrics:
            return {}
        
        return {
            'round_metrics_summary': {
                'total_rounds': len(self.round_metrics),
                'avg_accuracy': np.mean([rm.get('avg_accuracy', 0) for rm in self.round_metrics]),
                'final_accuracy': self.round_metrics[-1].get('avg_accuracy', 0),
                'best_accuracy': max(rm.get('avg_accuracy', 0) for rm in self.round_metrics)
            },
            'communication_summary': {
                'total_communication_time': self.total_communication_time,
                'total_training_time': self.total_training_time,
                'avg_comm_time_per_round': self.total_communication_time / len(self.round_metrics)
            },
            'experiment_metrics': self.experiment_metrics
        }
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export collected metrics to file.
        
        Args:
            filepath: Path to save metrics file
        """
        export_data = {
            'round_metrics': self.round_metrics,
            'experiment_metrics': self.experiment_metrics,
            'performance_history': self.performance_history,
            'summary': self.get_final_metrics(),
            'export_timestamp': time.time()
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            # Default to JSON if no extension
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.round_metrics = []
        self.experiment_metrics = {}
        self.communication_metrics = {}
        self.performance_history = []
        self.total_communication_time = 0.0
        self.total_training_time = 0.0
        self.total_aggregation_time = 0.0
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over rounds."""
        if not self.round_metrics:
            return {}
        
        return {
            'rounds': [rm.get('round_num', i) for i, rm in enumerate(self.round_metrics)],
            'accuracies': [rm.get('avg_accuracy', 0) for rm in self.round_metrics],
            'losses': [rm.get('avg_loss', 0) for rm in self.round_metrics],
            'participants': [rm.get('num_participants', 0) for rm in self.round_metrics]
        }
