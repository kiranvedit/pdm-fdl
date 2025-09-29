"""
Core Interfaces and Abstract Base Classes
=========================================
Defines the contract for all major components in the federated learning system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch


class ModelInterface(ABC):
    """Abstract interface for all federated learning models."""
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return model parameters as a dictionary."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from a dictionary."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass


class FederatedAlgorithmInterface(ABC):
    """Abstract interface for federated learning algorithms."""
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates into global model parameters."""
        pass
    
    @abstractmethod
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure algorithm-specific parameters for client training."""
        pass
    
    @abstractmethod
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update algorithm-specific state after aggregation."""
        pass


class DataSiteInterface(ABC):
    """Abstract interface for factory datasites."""
    
    @abstractmethod
    def initialize_datasite(self, config: Dict[str, Any]) -> None:
        """Initialize the PySyft datasite with configuration."""
        pass
    
    @abstractmethod
    def register_data(self, data: Dict[str, Any]) -> None:
        """Register local data as PySyft assets."""
        pass
    
    @abstractmethod
    def train_local_model(self, global_model: ModelInterface, 
                         training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train local model and return updates."""
        pass
    
    @abstractmethod
    def get_datasite_info(self) -> Dict[str, Any]:
        """Return datasite information and capabilities."""
        pass


class OrchestratorInterface(ABC):
    """Abstract interface for experiment orchestration."""
    
    @abstractmethod
    def setup_experiment(self, config: Dict[str, Any]) -> None:
        """Setup experiment with configuration."""
        pass
    
    @abstractmethod
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run a single federated learning experiment."""
        pass
    
    @abstractmethod
    def collect_results(self, experiment_id: str) -> Dict[str, Any]:
        """Collect and aggregate experiment results."""
        pass


class MetricsCollectorInterface(ABC):
    """Abstract interface for metrics collection."""
    
    @abstractmethod
    def collect_round_metrics(self, round_num: int, 
                            global_model: ModelInterface,
                            client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect metrics for a single round."""
        pass
    
    @abstractmethod
    def collect_experiment_metrics(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect overall experiment metrics."""
        pass
    
    @abstractmethod
    def export_metrics(self, filepath: str) -> None:
        """Export collected metrics to file."""
        pass


class CommunicationInterface(ABC):
    """Abstract interface for PySyft communication."""
    
    @abstractmethod
    def connect_to_datasite(self, datasite_url: str, credentials: Dict[str, str]) -> bool:
        """Connect to a factory datasite."""
        pass
    
    @abstractmethod
    def send_model(self, model_parameters: Dict[str, torch.Tensor], 
                   datasite_id: str) -> bool:
        """Send model parameters to a datasite."""
        pass
    
    @abstractmethod
    def request_training(self, training_config: Dict[str, Any], 
                        datasite_id: str) -> Dict[str, Any]:
        """Request local training from a datasite."""
        pass
    
    @abstractmethod
    def disconnect_from_datasite(self, datasite_id: str) -> bool:
        """Disconnect from a factory datasite."""
        pass
