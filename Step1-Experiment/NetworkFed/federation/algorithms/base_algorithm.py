"""
Base Federated Learning Algorithm Interface
===========================================
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch
from core.interfaces import FederatedAlgorithmInterface


class BaseFederatedAlgorithm(FederatedAlgorithmInterface):
    """
    Base implementation for federated learning algorithms.
    Provides common functionality and structure.
    """
    
    def __init__(self, learning_rate: float = 0.01, device: str = 'cpu'):
        self.learning_rate = learning_rate
        self.device = device
        self.round_num = 0
        self.algorithm_state = {}
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates into global model parameters.
        Default implementation uses weighted averaging.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate total samples for weighting
        total_samples = sum(update.get('num_samples', 1) for update in client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Weighted averaging of parameters
        # Track original dtypes to restore them after aggregation
        original_dtypes = {}
        
        for update in client_updates:
            weight = update.get('num_samples', 1) / total_samples
            model_update = update['model_update']
            
            for param_name, param_value in model_update.items():
                # Store original dtype for first parameter
                if param_name not in original_dtypes:
                    original_dtypes[param_name] = param_value.dtype
                
                if param_name not in aggregated_params:
                    # Initialize with float32 for aggregation
                    aggregated_params[param_name] = torch.zeros_like(param_value, dtype=torch.float32)
                
                # Convert parameter to float32 for aggregation
                param_value_float = param_value.float()
                aggregated_params[param_name] += weight * param_value_float
        
        # Restore original dtypes after aggregation
        for param_name in aggregated_params:
            if param_name in original_dtypes and original_dtypes[param_name] != torch.float32:
                aggregated_params[param_name] = aggregated_params[param_name].to(original_dtypes[param_name])
        
        return aggregated_params
    
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """
        Configure algorithm-specific parameters for client training.
        Base implementation provides common parameters.
        """
        return {
            'learning_rate': self.learning_rate,
            'round_num': round_num,
            'algorithm': self.__class__.__name__.lower()
        }
    
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """
        Update algorithm-specific state after aggregation.
        Base implementation updates round number.
        """
        self.round_num += 1
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the algorithm and its current state."""
        return {
            'algorithm_name': self.__class__.__name__,
            'round_num': self.round_num,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'state': self.algorithm_state.copy()
        }
    
    def reset_state(self) -> None:
        """Reset algorithm state to initial conditions."""
        self.round_num = 0
        self.algorithm_state = {}
    
    @abstractmethod
    def get_algorithm_specific_metrics(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get algorithm-specific metrics for monitoring."""
        pass
