#!/usr/bin/env python3
"""
Model Factory for Industrial Predictive Maintenance
==================================================
Creates model instances based on model type and configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
from .pdm_models import CentralCNNModel, CentralLSTMModel, CentralHybridModel


class ModelType:
    """Model type constants"""
    CNN = "cnn"
    LSTM = "lstm"
    HYBRID = "hybrid"


class ModelFactory:
    """
    Factory class for creating model instances.
    
    Supports CNN, LSTM, and Hybrid models for industrial predictive maintenance.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self.model_registry = {
            ModelType.CNN: CentralCNNModel,
            ModelType.LSTM: CentralLSTMModel,
            ModelType.HYBRID: CentralHybridModel
        }
    
    def create_model(self, model_type: Union[str, 'ModelType'], config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create (CNN, LSTM, HYBRID)
            config: Configuration dictionary for model parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        if config is None:
            config = {}
            
        # Handle both string and enum inputs
        if hasattr(model_type, 'value'):
            model_type_str = model_type.value
        else:
            model_type_str = str(model_type).lower()
        
        if model_type_str not in self.model_registry:
            raise ValueError(f"Unsupported model type: {model_type_str}. "
                           f"Supported types: {list(self.model_registry.keys())}")
        
        model_class = self.model_registry[model_type_str]
        
        # Extract model-specific parameters from config
        model_params = self._extract_model_params(model_type_str, config)
        
        # Create and return the model
        return model_class(**model_params)
    
    def _extract_model_params(self, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model-specific parameters from configuration.
        
        Args:
            model_type: Type of model
            config: Full configuration dictionary
            
        Returns:
            Model-specific parameters
        """
        # Default parameters for industrial predictive maintenance
        default_params = {
            'input_dim': config.get('input_dim', 5),  # 5 sensor features
            'num_classes': config.get('num_classes', 6),  # 6 failure types
        }
        
        if model_type == ModelType.CNN:
            default_params.update({
                'conv_filters': config.get('conv_filters', [32, 64, 128]),
                'fc_hidden': config.get('fc_hidden', [128, 64]),
                'dropout_rate': config.get('dropout_rate', 0.3)
            })
        
        elif model_type == ModelType.LSTM:
            default_params.update({
                'hidden_dim': config.get('hidden_size', 64),  # CentralLSTMModel uses hidden_dim
                'num_layers': config.get('num_layers', 2),
                'dropout_rate': config.get('dropout_rate', 0.3),
                'bidirectional': config.get('bidirectional', True)
            })
        
        elif model_type == ModelType.HYBRID:
            default_params.update({
                'cnn_filters': config.get('conv_filters', [32, 64]),  # CentralHybridModel uses cnn_filters
                'lstm_hidden': config.get('lstm_hidden_size', 64),    # CentralHybridModel uses lstm_hidden
                'dropout_rate': config.get('dropout_rate', 0.3)
            })
        
        return default_params
    
    def get_supported_models(self) -> list:
        """
        Get list of supported model types.
        
        Returns:
            List of supported model type strings
        """
        return list(self.model_registry.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in self.model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model_class = self.model_registry[model_type]
        
        return {
            'name': model_class.__name__,
            'type': model_type,
            'class': model_class,
            'default_params': self._extract_model_params(model_type, {})
        }


# Create a default instance for convenience
default_factory = ModelFactory()


def create_model(model_type: Union[str, 'ModelType'], config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Convenience function to create a model using the default factory.
    
    Args:
        model_type: Type of model to create
        config: Configuration dictionary
        
    Returns:
        Initialized model instance
    """
    return default_factory.create_model(model_type, config)
