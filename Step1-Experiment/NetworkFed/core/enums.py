"""
Enumerations for Federated Learning System
==========================================
"""

from enum import Enum, auto


class FederatedAlgorithm(Enum):
    """Supported federated learning algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDDYN = "feddyn"
    FEDNOVA = "fednova"


class ModelType(Enum):
    """Supported model architectures."""
    CNN = "cnn"
    LSTM = "lstm"
    HYBRID = "hybrid"


class DataDistribution(Enum):
    """Data distribution strategies."""
    IID = "iid"
    NON_IID_QUANTITY = "non_iid_quantity"
    NON_IID_FEATURE = "non_iid_feature"
    NON_IID_LABEL = "non_iid_label"


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class DataSiteStatus(Enum):
    """Factory datasite status."""
    OFFLINE = auto()
    ONLINE = auto()
    BUSY = auto()
    ERROR = auto()


class AggregationMethod(Enum):
    """Model aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    SIMPLE_AVERAGE = "simple_average"
    MEDIAN = "median"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    NONE = "none"
    BASIC = "basic"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
