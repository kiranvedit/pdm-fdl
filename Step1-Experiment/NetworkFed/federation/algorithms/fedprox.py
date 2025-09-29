"""
FedProx Algorithm Implementation
===============================
Federated learning with proximal terms for heterogeneous systems.
"""

from typing import Dict, List, Any
import torch
from .base_algorithm import BaseFederatedAlgorithm


class FedProxAlgorithm(BaseFederatedAlgorithm):
    """
    Federated Proximal (FedProx) algorithm implementation.
    
    FedProx adds a proximal term to handle system heterogeneity
    by adding regularization during local training.
    """
    
    def __init__(self, learning_rate: float = 0.01, mu: float = 0.01, device: str = 'cpu'):
        super().__init__(learning_rate, device)
        self.mu = mu  # Proximal term coefficient
        self.algorithm_state = {
            'total_rounds': 0,
            'mu': mu,
            'avg_proximal_loss': 0.0,
            'client_drift_metrics': []
        }
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.
        FedProx uses the same aggregation as FedAvg.
        """
        aggregated_params = super().aggregate(client_updates)
        self._update_fedprox_state(client_updates)
        return aggregated_params
    
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure FedProx-specific client training parameters."""
        config = super().configure_client_training(round_num)
        config.update({
            'algorithm': 'fedprox',
            'mu': self.mu,  # Proximal term coefficient
            'use_proximal_term': True,
            'local_epochs': 1
        })
        return config
    
    def get_algorithm_specific_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get FedProx-specific metrics."""
        if not client_updates:
            return {}
        
        # Calculate proximal regularization effects
        proximal_losses = [update.get('proximal_loss', 0) for update in client_updates]
        avg_proximal_loss = sum(proximal_losses) / len(proximal_losses) if proximal_losses else 0
        
        # Calculate client drift (difference from global model)
        client_drifts = [update.get('client_drift', 0) for update in client_updates]
        avg_client_drift = sum(client_drifts) / len(client_drifts) if client_drifts else 0
        max_client_drift = max(client_drifts) if client_drifts else 0
        
        return {
            'mu_coefficient': self.mu,
            'avg_proximal_loss': avg_proximal_loss,
            'avg_client_drift': avg_client_drift,
            'max_client_drift': max_client_drift,
            'num_participants': len(client_updates),
            'proximal_regularization_effect': self._calculate_regularization_effect(client_updates)
        }
    
    def _update_fedprox_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update FedProx-specific algorithm state."""
        # Update proximal loss tracking
        proximal_losses = [update.get('proximal_loss', 0) for update in client_updates]
        if proximal_losses:
            current_avg = sum(proximal_losses) / len(proximal_losses)
            rounds = self.algorithm_state['total_rounds']
            
            # Running average of proximal loss
            if rounds == 0:
                self.algorithm_state['avg_proximal_loss'] = current_avg
            else:
                self.algorithm_state['avg_proximal_loss'] = (
                    (self.algorithm_state['avg_proximal_loss'] * rounds + current_avg) / (rounds + 1)
                )
        
        # Track client drift metrics
        client_drifts = [update.get('client_drift', 0) for update in client_updates]
        if client_drifts:
            drift_stats = {
                'round': self.round_num,
                'avg_drift': sum(client_drifts) / len(client_drifts),
                'max_drift': max(client_drifts),
                'min_drift': min(client_drifts)
            }
            self.algorithm_state['client_drift_metrics'].append(drift_stats)
            
            # Keep only last 10 rounds of drift metrics
            if len(self.algorithm_state['client_drift_metrics']) > 10:
                self.algorithm_state['client_drift_metrics'] = \
                    self.algorithm_state['client_drift_metrics'][-10:]
        
        self.algorithm_state['total_rounds'] += 1
    
    def _calculate_regularization_effect(self, client_updates: List[Dict[str, Any]]) -> float:
        """
        Calculate the effect of proximal regularization.
        Returns a measure of how much the proximal term is constraining updates.
        """
        try:
            # Compare actual losses with what they would be without proximal term
            base_losses = [update.get('loss', 0) for update in client_updates]
            proximal_losses = [update.get('proximal_loss', 0) for update in client_updates]
            
            if not base_losses or not proximal_losses:
                return 0.0
            
            total_base = sum(base_losses)
            total_proximal = sum(proximal_losses)
            
            if total_base == 0:
                return 0.0
            
            # Regularization effect as percentage increase in loss
            effect = (total_proximal / total_base) if total_base > 0 else 0.0
            return min(effect, 10.0)  # Cap at 10x for numerical stability
            
        except Exception:
            return 0.0
    
    def update_mu(self, new_mu: float) -> None:
        """Update the proximal term coefficient."""
        self.mu = new_mu
        self.algorithm_state['mu'] = new_mu
    
    def get_client_drift_history(self) -> List[Dict[str, Any]]:
        """Get the history of client drift metrics."""
        return self.algorithm_state['client_drift_metrics'].copy()
    
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update algorithm state after aggregation."""
        super().update_algorithm_state(client_updates)
        self._update_fedprox_state(client_updates)
    
    def reset_state(self) -> None:
        """Reset FedProx state to initial conditions."""
        super().reset_state()
        self.algorithm_state = {
            'total_rounds': 0,
            'mu': self.mu,
            'avg_proximal_loss': 0.0,
            'client_drift_metrics': []
        }
