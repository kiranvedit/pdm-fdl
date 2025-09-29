"""
Model Factory for Creating Optimized Models
===========================================
Creates model instances based on your optimized configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Type
from core.interfaces import ModelInterface
from core.enums import ModelType
from core.exceptions import ModelError
from .optimized_models import OptimizedCNNModel, OptimizedLSTMModel, OptimizedHybridModel


class ModelWrapper(ModelInterface):
    """Wrapper to make existing models conform to ModelInterface."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return model parameters as a dictionary."""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from a dictionary."""
        model_dict = self.model.state_dict()
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name] = param
        self.model.load_state_dict(model_dict)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def __getattr__(self, name):
        """Delegate other method calls to the wrapped model."""
        return getattr(self.model, name)


class ModelFactory:
    """Factory for creating optimized model instances."""
    
    _model_registry = {
        ModelType.CNN: OptimizedCNNModel,
        ModelType.LSTM: OptimizedLSTMModel, 
        ModelType.HYBRID: OptimizedHybridModel
    }
    
    @classmethod
    def create_model(cls, model_type: ModelType, **kwargs) -> ModelInterface:
        """
        Create a model instance based on type and parameters.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            ModelInterface: Wrapped model instance
            
        Raises:
            ModelError: If model type is not supported
        """
        if model_type not in cls._model_registry:
            raise ModelError(f"Unsupported model type: {model_type}")
        
        model_class = cls._model_registry[model_type]
        
        try:
            # Create model with default or provided parameters
            model_instance = model_class(**kwargs)
            return ModelWrapper(model_instance)
        except Exception as e:
            raise ModelError(f"Failed to create {model_type.value} model: {str(e)}")
    
    @classmethod
    def get_default_config(cls, model_type: ModelType) -> Dict[str, Any]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dict with default configuration parameters
        """
        configs = {
            ModelType.CNN: {
                'input_dim': 10,
                'num_classes': 2,
                'conv_filters': [32, 64, 128],
                'fc_hidden': [256, 128],
                'dropout_rate': 0.3
            },
            ModelType.LSTM: {
                'input_dim': 10,
                'num_classes': 2,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.2,
                'bidirectional': True
            },
            ModelType.HYBRID: {
                'input_dim': 10,
                'num_classes': 2,
                'conv_filters': [32, 64],
                'lstm_hidden': 64,
                'fc_hidden': [128, 64],
                'dropout_rate': 0.25
            }
        }
        
        return configs.get(model_type, {})
    
    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type[nn.Module]) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Model type identifier
            model_class: Model class to register
        """
        cls._model_registry[model_type] = model_class
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model types."""
        return list(cls._model_registry.keys())


class ModelManager:
    """Manages model lifecycle and state."""
    
    def __init__(self):
        self.models = {}
        self.model_history = {}
    
    def create_and_register_model(self, model_id: str, model_type: ModelType, 
                                 **kwargs) -> ModelInterface:
        """
        Create and register a model with an ID.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model to create
            **kwargs: Model parameters
            
        Returns:
            ModelInterface: Created model instance
        """
        model = ModelFactory.create_model(model_type, **kwargs)
        self.models[model_id] = model
        self.model_history[model_id] = []
        return model
    
    def get_model(self, model_id: str) -> ModelInterface:
        """Get a registered model by ID."""
        if model_id not in self.models:
            raise ModelError(f"Model {model_id} not found")
        return self.models[model_id]
    
    def save_model_state(self, model_id: str, round_num: int = None) -> None:
        """Save current model state to history."""
        if model_id not in self.models:
            raise ModelError(f"Model {model_id} not found")
        
        state = self.models[model_id].get_parameters()
        self.model_history[model_id].append({
            'round': round_num,
            'state': state,
            'timestamp': torch.tensor(torch.initial_seed())  # Simple timestamp
        })
    
    def restore_model_state(self, model_id: str, round_num: int = None) -> None:
        """Restore model state from history."""
        if model_id not in self.model_history:
            raise ModelError(f"No history found for model {model_id}")
        
        history = self.model_history[model_id]
        if not history:
            raise ModelError(f"No saved states for model {model_id}")
        
        if round_num is None:
            # Restore latest state
            state = history[-1]['state']
        else:
            # Find specific round
            matching_states = [h for h in history if h['round'] == round_num]
            if not matching_states:
                raise ModelError(f"No state found for round {round_num}")
            state = matching_states[0]['state']
        
        self.models[model_id].set_parameters(state)
    
    def list_models(self) -> list:
        """List all registered model IDs."""
        return list(self.models.keys())
    
    def remove_model(self, model_id: str) -> None:
        """Remove a model and its history."""
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.model_history:
            del self.model_history[model_id]
