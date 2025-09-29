"""
FedDyn Algorithm Implementation
==============================
Federated learning with dynamic regularization for improved convergence.
"""

from typing import Dict, List, Any
import torch
import numpy as np
from .base_algorithm import BaseFederatedAlgorithm


class FedDynAlgorithm(BaseFederatedAlgorithm):
    """
    Federated Dynamic (FedDyn) algorithm implementation.
    
    FedDyn adds dynamic regularization to handle client drift and improve
    convergence in heterogeneous federated learning settings.
    """
    
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.01, device: str = 'cpu'):
        super().__init__(learning_rate, device)
        # Clamp alpha to prevent metrics explosion
        self.alpha = max(0.001, min(alpha, 0.1))  # Limit alpha between 0.001 and 0.1
        self.algorithm_state = {
            'total_rounds': 0,
            'alpha': self.alpha,
            'h_global': None,  # Global regularization parameter
            'client_h_states': {},  # Client-specific h states
            'gradient_diversity': 0.0,
            'regularization_history': []
        }
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FedDyn's dynamic regularization.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate total samples for weighting
        total_samples = sum(update.get('num_samples', 1) for update in client_updates)
        
        # Initialize aggregated parameters with proper type handling
        aggregated_params = {}
        original_dtypes = {}
        h_update = {}  # For FedDyn dynamic regularization
        
        # Standard weighted averaging with type safety
        for update in client_updates:
            weight = update.get('num_samples', 1) / total_samples
            model_update = update['model_update']
            
            for param_name, param_value in model_update.items():
                # Store original dtype
                if param_name not in original_dtypes:
                    original_dtypes[param_name] = param_value.dtype
                
                if param_name not in aggregated_params:
                    # Initialize with float32 for aggregation
                    aggregated_params[param_name] = torch.zeros_like(param_value, dtype=torch.float32)
                
                # Convert to float32 for aggregation
                param_value_float = param_value.float()
                aggregated_params[param_name] += weight * param_value_float
        
        # Restore original dtypes after aggregation
        for param_name in aggregated_params:
            if param_name in original_dtypes and original_dtypes[param_name] != torch.float32:
                aggregated_params[param_name] = aggregated_params[param_name].to(original_dtypes[param_name])
        
        # Update global h parameter (dynamic regularization term)
        self._update_global_h(client_updates, aggregated_params)
        
        # Apply FedDyn correction
        corrected_params = self._apply_feddyn_correction(aggregated_params)
        
        return corrected_params
    
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure FedDyn-specific client training parameters."""
        config = super().configure_client_training(round_num)
        config.update({
            'algorithm': 'feddyn',
            'alpha': self.alpha,
            'use_dynamic_regularization': True,
            'h_global': self.algorithm_state.get('h_global', {}),
            'local_epochs': 1
        })
        return config
    
    def _update_global_h(self, client_updates: List[Dict[str, Any]], 
                        aggregated_params: Dict[str, torch.Tensor]) -> None:
        """Update the global dynamic regularization parameter h."""
        
        # Initialize h_global if first round
        if self.algorithm_state['h_global'] is None:
            self.algorithm_state['h_global'] = {}
            for param_name in aggregated_params:
                # Always initialize h_global with float32 for arithmetic operations
                self.algorithm_state['h_global'][param_name] = torch.zeros_like(aggregated_params[param_name], dtype=torch.float32)
        
        # Calculate gradient diversity (measure of client heterogeneity)
        gradient_diversity = self._calculate_gradient_diversity(client_updates)
        self.algorithm_state['gradient_diversity'] = gradient_diversity
        
        # Update h based on client updates and gradient diversity
        for update in client_updates:
            client_id = update.get('client_id', f"client_{len(self.algorithm_state['client_h_states'])}")
            model_update = update['model_update']
            weight = update.get('num_samples', 1)
            
            # Update client-specific h state
            if client_id not in self.algorithm_state['client_h_states']:
                self.algorithm_state['client_h_states'][client_id] = {}
                for param_name, param_value in model_update.items():
                    # Always use float32 for client h_state to match global h
                    self.algorithm_state['client_h_states'][client_id][param_name] = torch.zeros_like(param_value, dtype=torch.float32)
            
            # FedDyn h update rule with stabilization
            for param_name, param_value in model_update.items():
                if param_name in self.algorithm_state['h_global']:
                    # Convert all tensors to float32 for arithmetic operations
                    param_value_float = param_value.float()
                    aggregated_float = aggregated_params[param_name].float()
                    
                    # Calculate h_delta with gradient clipping to prevent explosion
                    param_diff = aggregated_float - param_value_float
                    param_diff_norm = torch.norm(param_diff).item()
                    
                    # Clip large gradients to prevent instability
                    if param_diff_norm > 10.0:  # Gradient clipping threshold
                        param_diff = param_diff * (10.0 / param_diff_norm)
                    
                    # Update global h with stabilized delta (h_global is already float32)
                    h_delta = -self.alpha * param_diff
                    
                    # Additional norm clipping for h_delta
                    h_delta_norm = torch.norm(h_delta).item()
                    if h_delta_norm > 5.0:  # H delta clipping threshold
                        h_delta = h_delta * (5.0 / h_delta_norm)
                    
                    # Safe arithmetic: both h_global and h_delta are float32
                    self.algorithm_state['h_global'][param_name] += h_delta
                    
                    # Clip h_global to prevent accumulation explosion
                    h_global_norm = torch.norm(self.algorithm_state['h_global'][param_name]).item()
                    if h_global_norm > 50.0:  # Global h clipping threshold
                        self.algorithm_state['h_global'][param_name] = self.algorithm_state['h_global'][param_name] * (50.0 / h_global_norm)
                    
                    # Update client h state (also float32)
                    self.algorithm_state['client_h_states'][client_id][param_name] = h_delta
    
    def _apply_feddyn_correction(self, aggregated_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply FedDyn correction to aggregated parameters with stabilization."""
        corrected_params = {}
        
        for param_name, param_value in aggregated_params.items():
            if (self.algorithm_state['h_global'] is not None and 
                param_name in self.algorithm_state['h_global']):
                
                # Apply FedDyn correction: θ_t+1 = θ_agg - η * h_global
                # h_global is always float32, convert param_value for arithmetic
                h_global_param = self.algorithm_state['h_global'][param_name]  # Already float32
                param_value_float = param_value.float()
                
                # Calculate correction with stabilization
                correction = self.learning_rate * h_global_param
                
                # Clip correction to prevent parameter explosion
                correction_norm = torch.norm(correction).item()
                param_norm = torch.norm(param_value_float).item()
                
                # Adaptive correction clipping based on parameter magnitude
                max_correction = max(0.1 * param_norm, 1.0)  # At most 10% of param norm or 1.0
                if correction_norm > max_correction:
                    correction = correction * (max_correction / correction_norm)
                
                # Apply correction in float space
                corrected_param_float = param_value_float - correction
                
                # Convert back to original dtype
                corrected_param = corrected_param_float.to(param_value.dtype)
                
                # Final sanity check: clip if corrected parameters become too large
                corrected_norm = torch.norm(corrected_param.float()).item()
                if corrected_norm > 100.0:  # Hard limit to prevent explosion
                    corrected_param = corrected_param * (100.0 / corrected_norm)
                
                corrected_params[param_name] = corrected_param
            else:
                corrected_params[param_name] = param_value
        
        return corrected_params
    
    def _calculate_gradient_diversity(self, client_updates: List[Dict[str, Any]]) -> float:
        """
        Calculate gradient diversity as a measure of client heterogeneity.
        Returns a value indicating how diverse the client updates are.
        """
        if len(client_updates) < 2:
            return 0.0
        
        try:
            # Calculate pairwise gradient differences
            total_diversity = 0.0
            num_pairs = 0
            
            for i, update_i in enumerate(client_updates):
                for j, update_j in enumerate(client_updates[i+1:], i+1):
                    model_i = update_i['model_update']
                    model_j = update_j['model_update']
                    
                    # Calculate parameter-wise differences
                    param_differences = []
                    for param_name in model_i:
                        if param_name in model_j:
                            # Ensure both tensors are float for norm calculation
                            param_i = model_i[param_name].float()
                            param_j = model_j[param_name].float()
                            diff = torch.norm(param_i - param_j).item()
                            param_differences.append(diff)
                    
                    if param_differences:
                        avg_diff = sum(param_differences) / len(param_differences)
                        total_diversity += avg_diff
                        num_pairs += 1
            
            diversity = total_diversity / num_pairs if num_pairs > 0 else 0.0
            return min(diversity, 10.0)  # Cap for numerical stability
            
        except Exception:
            return 0.0
    
    def get_algorithm_specific_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get FedDyn-specific metrics with stabilization."""
        if not client_updates:
            return {}
        
        # Calculate regularization strength with numerical stability
        h_norms = []
        if self.algorithm_state['h_global']:
            for param_name, h_param in self.algorithm_state['h_global'].items():
                # Ensure h_param is float for norm calculation
                h_param_float = h_param.float() if h_param.dtype != torch.float32 else h_param
                h_norm = torch.norm(h_param_float).item()
                # Check for NaN/Inf and clamp extreme values
                if np.isfinite(h_norm):
                    h_norms.append(min(h_norm, 100.0))  # Cap individual norms
        
        avg_h_norm = np.mean(h_norms) if h_norms else 0.0
        
        # Ensure avg_h_norm is finite
        if not np.isfinite(avg_h_norm):
            avg_h_norm = 0.0
        
        # Track regularization history with bounds checking
        reg_strength = avg_h_norm * self.alpha
        reg_strength = min(reg_strength, 10.0)  # Cap regularization strength
        
        if np.isfinite(reg_strength):
            self.algorithm_state['regularization_history'].append(reg_strength)
        
        # Keep only last 10 rounds for memory efficiency
        if len(self.algorithm_state['regularization_history']) > 10:
            self.algorithm_state['regularization_history'] = \
                self.algorithm_state['regularization_history'][-10:]
        
        # Ensure gradient diversity is finite
        gradient_diversity = self.algorithm_state.get('gradient_diversity', 0.0)
        if not np.isfinite(gradient_diversity):
            gradient_diversity = 0.0
            self.algorithm_state['gradient_diversity'] = 0.0
        
        return {
            'alpha_coefficient': self.alpha,
            'gradient_diversity': gradient_diversity,
            'avg_h_norm': avg_h_norm,
            'regularization_strength': reg_strength,
            'num_participants': len(client_updates),
            'convergence_indicator': self._calculate_convergence_indicator(client_updates)
        }
    
    def _calculate_convergence_indicator(self, client_updates: List[Dict[str, Any]]) -> float:
        """
        Calculate convergence indicator based on regularization stability.
        """
        if len(self.algorithm_state['regularization_history']) < 3:
            return 0.5  # Default value
        
        try:
            recent_reg = self.algorithm_state['regularization_history'][-3:]
            reg_variance = np.var(recent_reg)
            
            # Lower variance indicates better convergence
            convergence = max(0, min(1, 1 - reg_variance))
            return float(convergence)
            
        except Exception:
            return 0.5
    
    def update_alpha(self, new_alpha: float) -> None:
        """Update the dynamic regularization coefficient with bounds checking."""
        # Clamp new alpha to safe range to prevent metrics explosion
        self.alpha = max(0.001, min(new_alpha, 0.1))
        self.algorithm_state['alpha'] = self.alpha
    
    def get_regularization_history(self) -> List[float]:
        """Get the history of regularization strengths."""
        return self.algorithm_state['regularization_history'].copy()
    
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update algorithm state after aggregation."""
        super().update_algorithm_state(client_updates)
        self.algorithm_state['total_rounds'] += 1
    
    def reset_state(self) -> None:
        """Reset FedDyn state to initial conditions."""
        super().reset_state()
        self.algorithm_state = {
            'total_rounds': 0,
            'alpha': self.alpha,
            'h_global': None,
            'client_h_states': {},
            'gradient_diversity': 0.0,
            'regularization_history': []
        }
