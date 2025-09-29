"""
FedAvg Algorithm Implementation
==============================
Standard Federated Averaging algorithm.
"""

from typing import Dict, List, Any
import torch
from .base_algorithm import BaseFederatedAlgorithm


class FedAvgAlgorithm(BaseFederatedAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm implementation.
    
    This is the standard federated learning algorithm that performs
    weighted averaging of client model updates.
    """
    
    def __init__(self, learning_rate: float = 0.01, device: str = 'cpu'):
        super().__init__(learning_rate, device)
        self.algorithm_state = {
            'total_rounds': 0,
            'total_clients_participated': 0,
            'average_client_samples': 0
        }
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.
        This is the core FedAvg aggregation.
        """
        # Use the base class weighted averaging implementation
        aggregated_params = super().aggregate(client_updates)
        
        # Update algorithm-specific state
        self._update_fedavg_state(client_updates)
        
        return aggregated_params
    
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure FedAvg-specific client training parameters."""
        config = super().configure_client_training(round_num)
        config.update({
            'algorithm': 'fedavg',
            'local_epochs': 1,  # Standard FedAvg uses 1 local epoch
            'use_proximal_term': False
        })
        return config
    
    def get_algorithm_specific_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get FedAvg-specific metrics."""
        if not client_updates:
            return {}
        
        # Calculate participation statistics
        num_participants = len(client_updates)
        total_samples = sum(update.get('num_samples', 0) for update in client_updates)
        avg_samples_per_client = total_samples / num_participants if num_participants > 0 else 0
        
        # Calculate weight distribution (how evenly distributed are the client weights)
        weights = [update.get('num_samples', 0) / total_samples for update in client_updates]
        weight_variance = torch.var(torch.tensor(weights)).item() if weights else 0
        
        return {
            'participants': num_participants,
            'total_samples': total_samples,
            'avg_samples_per_client': avg_samples_per_client,
            'weight_variance': weight_variance,
            'fedavg_convergence_indicator': self._calculate_convergence_indicator(client_updates)
        }
    
    def _update_fedavg_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update FedAvg-specific algorithm state."""
        self.algorithm_state['total_clients_participated'] += len(client_updates)
        
        total_samples = sum(update.get('num_samples', 0) for update in client_updates)
        current_avg = self.algorithm_state['average_client_samples']
        rounds = self.algorithm_state['total_rounds']
        
        # Update running average of client samples
        self.algorithm_state['average_client_samples'] = int(
            (current_avg * rounds + total_samples) / (rounds + 1)
        ) if rounds >= 0 else total_samples
        
        self.algorithm_state['total_rounds'] += 1
    
    def _calculate_convergence_indicator(self, client_updates: List[Dict[str, Any]]) -> float:
        """
        Calculate a simple convergence indicator based on client update similarity.
        Returns a value between 0 (high variance) and 1 (low variance/convergence).
        """
        if len(client_updates) < 2:
            return 1.0  # Perfect convergence with single client
        
        try:
            # Extract loss values if available
            losses = [update.get('loss', 0) for update in client_updates]
            if all(loss == 0 for loss in losses):
                return 0.5  # No loss information available
            
            # Calculate coefficient of variation (normalized standard deviation)
            mean_loss = sum(losses) / len(losses)
            if mean_loss == 0:
                return 1.0
            
            variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
            std_loss = variance ** 0.5
            cv = std_loss / mean_loss
            
            # Convert to convergence indicator (lower CV = higher convergence)
            convergence = max(0, min(1, 1 - cv))
            return convergence
            
        except Exception:
            return 0.5  # Default value if calculation fails
    
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update algorithm state after aggregation."""
        super().update_algorithm_state(client_updates)
        self._update_fedavg_state(client_updates)
    
    def reset_state(self) -> None:
        """Reset FedAvg state to initial conditions."""
        super().reset_state()
        self.algorithm_state = {
            'total_rounds': 0,
            'total_clients_participated': 0,
            'average_client_samples': 0
        }
