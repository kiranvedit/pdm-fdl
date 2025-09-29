"""
Federation module initialization
"""

from .algorithms import BaseFederatedAlgorithm, FedAvgAlgorithm, FedProxAlgorithm

__all__ = [
    'BaseFederatedAlgorithm',
    'FedAvgAlgorithm', 
    'FedProxAlgorithm'
]
