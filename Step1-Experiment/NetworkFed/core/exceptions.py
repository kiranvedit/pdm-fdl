"""
Custom Exceptions for Federated Learning System
===============================================
"""


class FederatedLearningError(Exception):
    """Base exception for federated learning system."""
    pass


class ModelError(FederatedLearningError):
    """Exception raised for model-related errors."""
    pass


class DataSiteError(FederatedLearningError):
    """Exception raised for datasite-related errors."""
    pass


class CommunicationError(FederatedLearningError):
    """Exception raised for communication errors."""
    pass


class AggregationError(FederatedLearningError):
    """Exception raised for aggregation errors."""
    pass


class ExperimentError(FederatedLearningError):
    """Exception raised for experiment execution errors."""
    pass


class ConfigurationError(FederatedLearningError):
    """Exception raised for configuration errors."""
    pass


class SecurityError(FederatedLearningError):
    """Exception raised for security-related errors."""
    pass


class DataError(FederatedLearningError):
    """Exception raised for data-related errors."""
    pass
