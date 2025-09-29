"""
Early Stopping Mechanisms for Federated Learning
===============================================
Implements early stopping at both local training and global aggregation levels.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time


class LocalEarlyStopping:
    """
    Early stopping mechanism for local training at datasites.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 monitor: str = 'loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.history = []
        
    def should_stop(self, current_metrics: Dict[str, float], epoch: int) -> bool:
        """
        Check if local training should stop early.
        
        Args:
            current_metrics: Dictionary with current epoch metrics
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in current_metrics:
            return False
        
        current_value = current_metrics[self.monitor]
        self.history.append({
            'epoch': epoch,
            'value': current_value,
            'metrics': current_metrics.copy()
        })
        
        # First epoch
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check improvement
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Check if we should stop
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False
    
    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement over best value."""
        if self.mode == 'min':
            return current_value < self.best_value - self.min_delta
        elif self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            raise ValueError(f"Mode {self.mode} not supported. Use 'min' or 'max'.")
    
    def get_best_epoch(self) -> Optional[int]:
        """Get the epoch with the best value."""
        if not self.history:
            return None
        
        if self.mode == 'min':
            best_entry = min(self.history, key=lambda x: x['value'])
        else:
            best_entry = max(self.history, key=lambda x: x['value'])
        
        return best_entry['epoch']
    
    def reset(self):
        """Reset early stopping state."""
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.history = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of early stopping behavior."""
        return {
            'early_stopped': self.stopped_epoch > 0,
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.get_best_epoch(),
            'best_value': self.best_value,
            'patience': self.patience,
            'wait_count': self.wait_count,
            'monitor': self.monitor,
            'mode': self.mode
        }


class GlobalEarlyStopping:
    """
    Early stopping mechanism for global federated learning aggregation.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 monitor: str = 'accuracy', mode: str = 'max',
                 min_rounds: int = 5, convergence_threshold: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.min_rounds = min_rounds
        self.convergence_threshold = convergence_threshold
        
        self.best_value = None
        self.best_round = 0
        self.wait_count = 0
        self.stopped_round = 0
        self.round_history = []
        self.convergence_history = []
        
    def should_stop(self, global_metrics: Dict[str, float], round_num: int,
                   model_params: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[bool, str]:
        """
        Check if global federated learning should stop early.
        
        Args:
            global_metrics: Global evaluation metrics
            round_num: Current federated learning round
            model_params: Current global model parameters (for convergence check)
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Record round history
        self.round_history.append({
            'round': round_num,
            'metrics': global_metrics.copy(),
            'timestamp': time.time()
        })
        
        # Don't stop before minimum rounds
        if round_num < self.min_rounds:
            return False, "minimum_rounds_not_reached"
        
        # Check performance-based early stopping
        performance_stop, perf_reason = self._check_performance_stopping(global_metrics, round_num)
        if performance_stop:
            return True, perf_reason
        
        # Check convergence-based early stopping
        if model_params:
            convergence_stop, conv_reason = self._check_convergence_stopping(model_params, round_num)
            if convergence_stop:
                return True, conv_reason
        
        return False, "continue_training"
    
    def _check_performance_stopping(self, global_metrics: Dict[str, float], 
                                   round_num: int) -> Tuple[bool, str]:
        """Check performance-based early stopping."""
        if self.monitor not in global_metrics:
            return False, "monitor_metric_missing"
        
        current_value = global_metrics[self.monitor]
        
        # First round
        if self.best_value is None:
            self.best_value = current_value
            self.best_round = round_num
            return False, "first_evaluation"
        
        # Check improvement
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_round = round_num
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Check if we should stop
        if self.wait_count >= self.patience:
            self.stopped_round = round_num
            return True, f"no_improvement_for_{self.patience}_rounds"
        
        return False, "performance_improving"
    
    def _check_convergence_stopping(self, model_params: Dict[str, torch.Tensor],
                                   round_num: int) -> Tuple[bool, str]:
        """Check convergence-based early stopping."""
        # Calculate parameter change from previous round
        if len(self.convergence_history) > 0:
            prev_params = self.convergence_history[-1]['params']
            param_change = self._calculate_parameter_change(prev_params, model_params)
            
            # Record convergence info
            self.convergence_history.append({
                'round': round_num,
                'params': {name: param.clone() for name, param in model_params.items()},
                'param_change': param_change
            })
            
            # Check if converged
            if param_change < self.convergence_threshold:
                return True, f"converged_param_change_{param_change:.6f}"
        else:
            # First round with parameters
            self.convergence_history.append({
                'round': round_num,
                'params': {name: param.clone() for name, param in model_params.items()},
                'param_change': float('inf')
            })
        
        return False, "not_converged"
    
    def _calculate_parameter_change(self, prev_params: Dict[str, torch.Tensor],
                                   curr_params: Dict[str, torch.Tensor]) -> float:
        """Calculate the magnitude of parameter change between rounds."""
        total_change = 0.0
        total_params = 0
        
        for param_name in prev_params.keys():
            if param_name in curr_params:
                param_diff = curr_params[param_name] - prev_params[param_name]
                change = torch.norm(param_diff).item()
                total_change += change
                total_params += 1
        
        return total_change / total_params if total_params > 0 else 0.0
    
    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement over best value."""
        if self.mode == 'min':
            return current_value < self.best_value - self.min_delta
        elif self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            raise ValueError(f"Mode {self.mode} not supported. Use 'min' or 'max'.")
    
    def get_best_round(self) -> int:
        """Get the round with the best performance."""
        return self.best_round
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Get detailed convergence analysis."""
        if len(self.convergence_history) < 2:
            return {'status': 'insufficient_data'}
        
        param_changes = [entry['param_change'] for entry in self.convergence_history[1:]]
        
        return {
            'status': 'analyzed',
            'final_param_change': param_changes[-1] if param_changes else 0.0,
            'avg_param_change': np.mean(param_changes) if param_changes else 0.0,
            'param_change_trend': self._calculate_trend(param_changes),
            'convergence_threshold': self.convergence_threshold,
            'converged': param_changes[-1] < self.convergence_threshold if param_changes else False
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in parameter changes."""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -0.001:
            return 'decreasing'  # Converging
        elif slope > 0.001:
            return 'increasing'  # Diverging
        else:
            return 'stable'
    
    def reset(self):
        """Reset global early stopping state."""
        self.best_value = None
        self.best_round = 0
        self.wait_count = 0
        self.stopped_round = 0
        self.round_history = []
        self.convergence_history = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of global early stopping."""
        summary = {
            'early_stopped': self.stopped_round > 0,
            'stopped_round': self.stopped_round,
            'best_round': self.best_round,
            'best_value': self.best_value,
            'patience': self.patience,
            'wait_count': self.wait_count,
            'monitor': self.monitor,
            'mode': self.mode,
            'total_rounds': len(self.round_history),
            'convergence_analysis': self.get_convergence_analysis()
        }
        
        # Add performance history
        if self.round_history:
            performance_values = [r['metrics'].get(self.monitor, 0) for r in self.round_history]
            summary['performance_history'] = performance_values
            summary['performance_improvement'] = (
                performance_values[-1] - performance_values[0] if len(performance_values) > 1 else 0
            )
        
        return summary


class EarlyStoppingManager:
    """
    Manages both local and global early stopping mechanisms.
    """
    
    def __init__(self, local_config: Optional[Dict[str, Any]] = None,
                 global_config: Optional[Dict[str, Any]] = None):
        
        # Default configurations
        default_local = {
            'patience': 5,
            'min_delta': 0.001,
            'monitor': 'loss',
            'mode': 'min'
        }
        
        default_global = {
            'patience': 10,
            'min_delta': 0.001,
            'monitor': 'accuracy',
            'mode': 'max',
            'min_rounds': 5,
            'convergence_threshold': 0.0001
        }
        
        # Merge configurations
        local_config = {**default_local, **(local_config or {})}
        global_config = {**default_global, **(global_config or {})}
        
        # Create early stopping instances
        self.local_early_stopping = LocalEarlyStopping(**local_config)
        self.global_early_stopping = GlobalEarlyStopping(**global_config)
        
        # Tracking
        self.local_stops_per_round = {}
        self.experiment_summary = {}
    
    def create_local_early_stopping(self, datasite_id: str) -> LocalEarlyStopping:
        """Create a new local early stopping instance for a datasite."""
        # Create a copy of the local early stopping configuration
        local_es = LocalEarlyStopping(
            patience=self.local_early_stopping.patience,
            min_delta=self.local_early_stopping.min_delta,
            monitor=self.local_early_stopping.monitor,
            mode=self.local_early_stopping.mode
        )
        return local_es
    
    def check_local_stopping(self, datasite_id: str, metrics: Dict[str, float], 
                           epoch: int, local_es: LocalEarlyStopping) -> bool:
        """Check if local training should stop for a specific datasite."""
        should_stop = local_es.should_stop(metrics, epoch)
        
        if should_stop:
            # Track local stops
            round_key = f"round_{epoch // 10}"  # Approximate round grouping
            if round_key not in self.local_stops_per_round:
                self.local_stops_per_round[round_key] = []
            self.local_stops_per_round[round_key].append({
                'datasite_id': datasite_id,
                'epoch': epoch,
                'summary': local_es.get_summary()
            })
        
        return should_stop
    
    def check_global_stopping(self, global_metrics: Dict[str, float], round_num: int,
                            model_params: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[bool, str]:
        """Check if global federated learning should stop."""
        return self.global_early_stopping.should_stop(global_metrics, round_num, model_params)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of early stopping for the entire experiment."""
        return {
            'local_early_stopping': {
                'configuration': {
                    'patience': self.local_early_stopping.patience,
                    'min_delta': self.local_early_stopping.min_delta,
                    'monitor': self.local_early_stopping.monitor,
                    'mode': self.local_early_stopping.mode
                },
                'stops_per_round': self.local_stops_per_round
            },
            'global_early_stopping': {
                'configuration': {
                    'patience': self.global_early_stopping.patience,
                    'min_delta': self.global_early_stopping.min_delta,
                    'monitor': self.global_early_stopping.monitor,
                    'mode': self.global_early_stopping.mode,
                    'min_rounds': self.global_early_stopping.min_rounds,
                    'convergence_threshold': self.global_early_stopping.convergence_threshold
                },
                'summary': self.global_early_stopping.get_summary()
            }
        }
    
    def reset_all(self):
        """Reset all early stopping mechanisms."""
        self.local_early_stopping.reset()
        self.global_early_stopping.reset()
        self.local_stops_per_round = {}
        self.experiment_summary = {}
