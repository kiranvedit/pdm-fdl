"""
NetworkFed - Modular PySyft Federated Learning System
====================================================

A clean, modular architecture for real federated learning experiments using PySyft.

Core Components:
- Core: Interfaces, enums, and exceptions
- Models: Model factory and optimized model definitions  
- Federation: Algorithms, aggregation, and communication
- Orchestration: Experiment orchestration and configuration
- DataSite: Factory datasite implementation
- Monitoring: Metrics collection and performance tracking
- Security: Privacy and security mechanisms
- Utils: Utilities and helpers
"""

# Core components
from .core import (
    ModelInterface,
    FederatedAlgorithmInterface,
    DataSiteInterface,
    OrchestratorInterface,
    MetricsCollectorInterface,
    CommunicationInterface,
    FederatedAlgorithm,
    ModelType,
    DataDistribution,
    ExperimentStatus,
    DataSiteStatus,
    AggregationMethod,
    PrivacyLevel
)

# Models
from .models import (
    ModelFactory,
    ModelManager,
    OptimizedCNNModel,
    OptimizedLSTMModel,
    OptimizedHybridModel
)

# Federation algorithms
from .federation.algorithms import (
    BaseFederatedAlgorithm,
    FedAvgAlgorithm,
    FedProxAlgorithm
)

# Communication
from .federation.communication.syft_client import PySyftCommunicationManager

# Orchestration
from .orchestration.orchestrator import FederatedExperimentOrchestrator
from .orchestration.experiment_config import ExperimentConfig, ConfigurationManager

# DataSite
from .datasite.factory_node import FactoryDataSite

# Monitoring
from .monitoring.metrics_collector import MetricsCollector

__version__ = "1.0.0"
__author__ = "PDM-FDL Research Team"

__all__ = [
    # Core interfaces and enums
    'ModelInterface',
    'FederatedAlgorithmInterface', 
    'DataSiteInterface',
    'OrchestratorInterface',
    'MetricsCollectorInterface',
    'CommunicationInterface',
    'FederatedAlgorithm',
    'ModelType',
    'DataDistribution',
    'ExperimentStatus',
    'DataSiteStatus',
    'AggregationMethod',
    'PrivacyLevel',
    
    # Models
    'ModelFactory',
    'ModelManager',
    'OptimizedCNNModel',
    'OptimizedLSTMModel', 
    'OptimizedHybridModel',
    
    # Federation
    'BaseFederatedAlgorithm',
    'FedAvgAlgorithm',
    'FedProxAlgorithm',
    'PySyftCommunicationManager',
    
    # Orchestration
    'FederatedExperimentOrchestrator',
    'ExperimentConfig',
    'ConfigurationManager',
    
    # DataSite
    'FactoryDataSite',
    
    # Monitoring
    'MetricsCollector'
]
