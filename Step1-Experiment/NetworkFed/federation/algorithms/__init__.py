"""
Algorithm module initialization
"""

from .base_algorithm import BaseFederatedAlgorithm
from .fedavg import FedAvgAlgorithm  
from .fedprox import FedProxAlgorithm
from .feddyn import FedDynAlgorithm
from .fednova import FedNovaAlgorithm

# Available algorithms registry
AVAILABLE_ALGORITHMS = {
    'fedavg': FedAvgAlgorithm,
    'fedprox': FedProxAlgorithm,
    'feddyn': FedDynAlgorithm,
    'fednova': FedNovaAlgorithm
}

__all__ = [
    'BaseFederatedAlgorithm',
    'FedAvgAlgorithm',
    'FedProxAlgorithm',
    'FedDynAlgorithm',
    'FedNovaAlgorithm',
    'AVAILABLE_ALGORITHMS'
]
