"""
Core module initialization
"""

from .interfaces import (
    ModelInterface,
    FederatedAlgorithmInterface, 
    DataSiteInterface,
    OrchestratorInterface,
    MetricsCollectorInterface,
    CommunicationInterface
)

from .enums import (
    FederatedAlgorithm,
    ModelType,
    DataDistribution,
    ExperimentStatus,
    DataSiteStatus,
    AggregationMethod,
    PrivacyLevel
)

from .exceptions import (
    FederatedLearningError,
    ModelError,
    DataSiteError,
    CommunicationError,
    AggregationError,
    ExperimentError,
    ConfigurationError,
    SecurityError,
    DataError
)

__all__ = [
    # Interfaces
    'ModelInterface',
    'FederatedAlgorithmInterface',
    'DataSiteInterface', 
    'OrchestratorInterface',
    'MetricsCollectorInterface',
    'CommunicationInterface',
    
    # Enums
    'FederatedAlgorithm',
    'ModelType',
    'DataDistribution',
    'ExperimentStatus',
    'DataSiteStatus',
    'AggregationMethod',
    'PrivacyLevel',
    
    # Exceptions
    'FederatedLearningError',
    'ModelError',
    'DataSiteError',
    'CommunicationError',
    'AggregationError',
    'ExperimentError',
    'ConfigurationError',
    'SecurityError',
    'DataError'
]
