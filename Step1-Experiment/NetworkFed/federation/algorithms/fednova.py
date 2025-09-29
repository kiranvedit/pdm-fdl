"""
FedNova Algorithm Implementation
===============================
Federated learning with normalized averaging for handling client heterogeneity.
"""

from typing import Dict, List, Any
import torch
import numpy as np
from .base_algorithm import BaseFederatedAlgorithm


class FedNovaAlgorithm(BaseFederatedAlgorithm):
    """
    Federated Nova (FedNova) algorithm implementation.
    
    FedNova addresses objective inconsistency in federated learning by 
    normalizing client updates based on their local training intensity.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, device: str = 'cpu'):
        super().__init__(learning_rate, device)
        self.momentum = momentum
        self.algorithm_state = {
            'total_rounds': 0,
            'momentum_buffer': None,
            'client_tau_eff': {},  # Effective local epochs per client
            'global_tau_eff': 0.0,  # Global effective tau
            'variance_reduction_factor': 1.0,
            'normalization_history': []
        }
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FedNova's normalized averaging.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate effective tau (local training intensity) for each client
        self._calculate_effective_tau(client_updates)
        
        # Normalize client updates based on their effective tau
        normalized_updates = self._normalize_client_updates(client_updates)
        
        # Apply weighted averaging with variance reduction
        aggregated_params = self._variance_reduced_averaging(normalized_updates)
        
        # Apply momentum if momentum buffer exists
        aggregated_params = self._apply_momentum(aggregated_params)
        
        return aggregated_params
    
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure FedNova-specific client training parameters."""
        config = super().configure_client_training(round_num)
        config.update({
            'algorithm': 'fednova',
            'track_tau_eff': True,
            'momentum': self.momentum,
            'variance_reduction': True,
            'local_epochs': 1  # FedNova typically uses single epoch per round
        })
        return config
    
    def _calculate_effective_tau(self, client_updates: List[Dict[str, Any]]) -> None:
        """
        Calculate effective tau (local training intensity) for each client.
        
        Tau_eff measures how much actual training each client performed,
        accounting for data heterogeneity and local training variations.
        """
        total_samples = sum(update.get('num_samples', 1) for update in client_updates)
        total_samples = max(total_samples, 1)  # Prevent division by zero
        
        for update in client_updates:
            client_id = update['client_id']
            num_samples = update.get('num_samples', 1)
            local_epochs = update.get('local_epochs', 1)
            
            # Calculate effective tau based on sample size and training intensity
            # This is a simplified version - in practice, tau_eff should be computed
            # based on the actual gradient norms and convergence behavior
            base_tau = local_epochs
            sample_weight = num_samples / total_samples
            
            # Adjust tau based on training effectiveness
            training_loss = update.get('training_loss', 1.0)
            loss_improvement = update.get('loss_improvement', 0.1)
            
            # Tau_eff reflects actual training progress
            # Handle division by zero when training_loss is very small or zero
            if loss_improvement > 0 and training_loss > 1e-8:
                effectiveness_factor = min(2.0, loss_improvement / training_loss)
            elif training_loss <= 1e-8:
                # Perfect or near-perfect training (loss ~= 0)
                effectiveness_factor = 2.0 if loss_improvement > 0 else 1.0
            else:
                effectiveness_factor = 0.5
            
            tau_eff = base_tau * effectiveness_factor * sample_weight
            self.algorithm_state['client_tau_eff'][client_id] = max(0.1, tau_eff)
        
        # Calculate global effective tau
        client_taus = list(self.algorithm_state['client_tau_eff'].values())
        self.algorithm_state['global_tau_eff'] = np.mean(client_taus) if client_taus else 1.0
    
    def _normalize_client_updates(self, client_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize client updates based on their effective tau values.
        """
        normalized_updates = []
        global_tau = self.algorithm_state['global_tau_eff']
        
        for update in client_updates:
            client_id = update['client_id']
            tau_eff = self.algorithm_state['client_tau_eff'].get(client_id, 1.0)
            
            # Normalization factor: tau_eff / global_tau_eff
            normalization_factor = tau_eff / max(global_tau, 0.1)
            
            # Normalize model updates
            normalized_model_update = {}
            for param_name, param_value in update['model_update'].items():
                normalized_model_update[param_name] = param_value * normalization_factor
            
            normalized_update = update.copy()
            normalized_update['model_update'] = normalized_model_update
            normalized_update['normalization_factor'] = normalization_factor
            
            normalized_updates.append(normalized_update)
        
        return normalized_updates
    
    def _variance_reduced_averaging(self, normalized_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Apply variance-reduced weighted averaging to normalized updates.
        """
        if not normalized_updates:
            return {}
        
        # Calculate weights based on sample sizes and normalization factors
        total_weight = 0.0
        weighted_params = {}
        original_dtypes = {}
        
        for update in normalized_updates:
            num_samples = update.get('num_samples', 1)
            norm_factor = update.get('normalization_factor', 1.0)
            
            # Weight combines sample size and normalization quality
            weight = num_samples * norm_factor
            total_weight += weight
            
            model_update = update['model_update']
            
            for param_name, param_value in model_update.items():
                # Store original dtype
                if param_name not in original_dtypes:
                    original_dtypes[param_name] = param_value.dtype
                
                if param_name not in weighted_params:
                    # Initialize with float32 for aggregation
                    weighted_params[param_name] = torch.zeros_like(param_value, dtype=torch.float32)
                
                # Convert to float32 for aggregation
                param_value_float = param_value.float()
                weighted_params[param_name] += weight * param_value_float
        
        # Normalize by total weight and restore original dtypes
        aggregated_params = {}
        for param_name, param_value in weighted_params.items():
            normalized_param = param_value / max(total_weight, 1e-10)
            
            # Restore original dtype
            if param_name in original_dtypes and original_dtypes[param_name] != torch.float32:
                normalized_param = normalized_param.to(original_dtypes[param_name])
            
            aggregated_params[param_name] = normalized_param
        
        # Update variance reduction factor
        self._update_variance_reduction_factor(normalized_updates)
        
        return aggregated_params
    
    def _apply_momentum(self, aggregated_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply momentum to the aggregated parameters.
        """
        if self.momentum == 0.0:
            return aggregated_params
        
        # Initialize momentum buffer if first round
        if self.algorithm_state['momentum_buffer'] is None:
            self.algorithm_state['momentum_buffer'] = {}
            for param_name, param_value in aggregated_params.items():
                self.algorithm_state['momentum_buffer'][param_name] = torch.zeros_like(param_value)
        
        # Apply momentum: v_t = momentum * v_{t-1} + (1 - momentum) * gradient
        momentum_params = {}
        for param_name, param_value in aggregated_params.items():
            if param_name in self.algorithm_state['momentum_buffer']:
                momentum_update = (self.momentum * self.algorithm_state['momentum_buffer'][param_name] + 
                                 (1 - self.momentum) * param_value)
                self.algorithm_state['momentum_buffer'][param_name] = momentum_update
                momentum_params[param_name] = momentum_update
            else:
                # Initialize new parameter in momentum buffer
                self.algorithm_state['momentum_buffer'][param_name] = param_value
                momentum_params[param_name] = param_value
        
        return momentum_params
    
    def _update_variance_reduction_factor(self, normalized_updates: List[Dict[str, Any]]) -> None:
        """
        Update variance reduction factor based on update consistency.
        """
        if len(normalized_updates) < 2:
            self.algorithm_state['variance_reduction_factor'] = 1.0
            return
        
        try:
            # Calculate variance across normalized updates
            param_variances = []
            
            # Get parameter names from first update
            first_update = normalized_updates[0]['model_update']
            
            for param_name in first_update:
                param_values = []
                for update in normalized_updates:
                    if param_name in update['model_update']:
                        param_norm = torch.norm(update['model_update'][param_name]).item()
                        param_values.append(param_norm)
                
                if len(param_values) > 1:
                    param_variance = np.var(param_values)
                    param_variances.append(param_variance)
            
            if param_variances:
                avg_variance = np.mean(param_variances)
                # Variance reduction factor decreases with higher variance
                self.algorithm_state['variance_reduction_factor'] = 1.0 / (1.0 + avg_variance)
            else:
                self.algorithm_state['variance_reduction_factor'] = 1.0
                
        except Exception:
            self.algorithm_state['variance_reduction_factor'] = 1.0
    
    def get_algorithm_specific_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get FedNova-specific metrics."""
        if not client_updates:
            return {}
        
        # Calculate normalization statistics
        normalization_factors = []
        tau_eff_values = list(self.algorithm_state['client_tau_eff'].values())
        
        for update in client_updates:
            client_id = update['client_id']
            tau_eff = self.algorithm_state['client_tau_eff'].get(client_id, 1.0)
            global_tau = self.algorithm_state['global_tau_eff']
            norm_factor = tau_eff / max(global_tau, 0.1)
            normalization_factors.append(norm_factor)
        
        # Track normalization quality
        normalization_quality = {
            'mean_normalization': np.mean(normalization_factors) if normalization_factors else 1.0,
            'std_normalization': np.std(normalization_factors) if normalization_factors else 0.0,
            'variance_reduction': self.algorithm_state['variance_reduction_factor']
        }
        
        self.algorithm_state['normalization_history'].append(normalization_quality)
        
        # Keep only last 10 rounds for memory efficiency
        if len(self.algorithm_state['normalization_history']) > 10:
            self.algorithm_state['normalization_history'] = \
                self.algorithm_state['normalization_history'][-10:]
        
        return {
            'momentum': self.momentum,
            'global_tau_eff': self.algorithm_state['global_tau_eff'],
            'mean_client_tau_eff': np.mean(tau_eff_values) if tau_eff_values else 1.0,
            'std_client_tau_eff': np.std(tau_eff_values) if tau_eff_values else 0.0,
            'mean_normalization_factor': normalization_quality['mean_normalization'],
            'normalization_consistency': 1.0 - normalization_quality['std_normalization'],
            'variance_reduction_factor': self.algorithm_state['variance_reduction_factor'],
            'num_participants': len(client_updates),
            'convergence_stability': self._calculate_convergence_stability()
        }
    
    def _calculate_convergence_stability(self) -> float:
        """
        Calculate convergence stability based on normalization consistency.
        """
        if len(self.algorithm_state['normalization_history']) < 3:
            return 0.5  # Default value
        
        try:
            recent_history = self.algorithm_state['normalization_history'][-3:]
            variance_reductions = [h['variance_reduction'] for h in recent_history]
            
            # Stability measured by consistency of variance reduction
            stability = 1.0 - np.std(variance_reductions)
            return float(max(0, min(1, stability)))
            
        except Exception:
            return 0.5
    
    def update_momentum(self, new_momentum: float) -> None:
        """Update the momentum coefficient."""
        self.momentum = new_momentum
    
    def get_normalization_history(self) -> List[Dict[str, float]]:
        """Get the history of normalization statistics."""
        return self.algorithm_state['normalization_history'].copy()
    
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update algorithm state after aggregation."""
        super().update_algorithm_state(client_updates)
        self.algorithm_state['total_rounds'] += 1
    
    def reset_state(self) -> None:
        """Reset FedNova state to initial conditions."""
        super().reset_state()
        self.algorithm_state = {
            'total_rounds': 0,
            'momentum_buffer': None,
            'client_tau_eff': {},
            'global_tau_eff': 0.0,
            'variance_reduction_factor': 1.0,
            'normalization_history': []
        }
