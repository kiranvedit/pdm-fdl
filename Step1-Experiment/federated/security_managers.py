"""
Security Managers for Federated Learning
Contains differential privacy and Byzantine fault tolerance implementations
"""
import torch
import numpy as np
from typing import List, Set, Dict, Any
from scipy import stats


class DifferentialPrivacyManager:
    """Differential Privacy Manager for Federated Learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0, noise_multiplier: float = 1.0):
        """
        Initialize Differential Privacy Manager
        
        Args:
            epsilon: Privacy budget
            delta: Delta parameter for (ε,δ)-differential privacy
            sensitivity: Global sensitivity of the function
            noise_multiplier: Multiplier for noise scale
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_multiplier = noise_multiplier
        
    def add_noise_to_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add calibrated noise to gradients for differential privacy"""
        noisy_gradients = []
        
        # Calculate noise scale based on privacy parameters
        sigma = (self.sensitivity * self.noise_multiplier) / self.epsilon
        
        for grad in gradients:
            if grad is not None:
                # Add Gaussian noise
                noise = torch.normal(0, sigma, size=grad.shape, device=grad.device)
                noisy_grad = grad + noise
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(None)
                
        return noisy_gradients
    
    def add_noise_to_model_update(self, model_update: torch.Tensor) -> torch.Tensor:
        """Add noise to model update"""
        sigma = (self.sensitivity * self.noise_multiplier) / self.epsilon
        noise = torch.normal(0, sigma, size=model_update.shape, device=model_update.device)
        return model_update + noise
    
    def clip_gradients(self, gradients: List[torch.Tensor], max_norm: float = 1.0) -> List[torch.Tensor]:
        """Clip gradients to bound sensitivity"""
        clipped_gradients = []
        
        for grad in gradients:
            if grad is not None:
                grad_norm = torch.norm(grad)
                if grad_norm > max_norm:
                    clipped_grad = grad * (max_norm / grad_norm)
                else:
                    clipped_grad = grad
                clipped_gradients.append(clipped_grad)
            else:
                clipped_gradients.append(None)
                
        return clipped_gradients
    
    def get_privacy_spent(self, num_rounds: int) -> tuple:
        """Calculate privacy spent after num_rounds"""
        # Simple composition (can be improved with advanced composition)
        epsilon_spent = num_rounds * self.epsilon
        delta_spent = num_rounds * self.delta
        return epsilon_spent, delta_spent


class ByzantineFaultTolerance:
    """Byzantine Fault Tolerance for Federated Learning"""
    
    def __init__(self, anomaly_threshold: float = 2.0, min_clients: int = 3):
        """
        Initialize Byzantine Fault Tolerance
        
        Args:
            anomaly_threshold: Z-score threshold for anomaly detection
            min_clients: Minimum number of clients required
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_clients = min_clients
        
    def detect_byzantine_clients(self, client_ids: List[str],
                               update_tensors: List[torch.Tensor]) -> Set[str]:
        """Detect Byzantine clients based on cosine similarity analysis"""
        if len(update_tensors) < 2:
            return set()
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i, tensor_i in enumerate(update_tensors):
            client_similarities = []
            for j, tensor_j in enumerate(update_tensors):
                if i != j:
                    cos_sim = torch.cosine_similarity(tensor_i.unsqueeze(0), tensor_j.unsqueeze(0))
                    client_similarities.append(cos_sim.item())
            similarities.append(np.mean(client_similarities))
        
        # Identify outliers using z-score
        z_scores = np.abs(stats.zscore(similarities))
        byzantine_indices = np.where(z_scores > self.anomaly_threshold)[0]
        
        return {client_ids[i] for i in byzantine_indices}
    
    def aggregate_with_trimmed_mean(self, updates: List[torch.Tensor], 
                                  trim_ratio: float = 0.2) -> torch.Tensor:
        """Aggregate updates using trimmed mean to handle Byzantine clients"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Stack all updates
        stacked = torch.stack(updates)
        
        # Calculate trimmed mean
        n_trim = int(len(updates) * trim_ratio)
        if n_trim > 0:
            sorted_updates, _ = torch.sort(stacked, dim=0)
            trimmed_updates = sorted_updates[n_trim:-n_trim] if n_trim < len(updates)//2 else sorted_updates
            aggregated = torch.mean(trimmed_updates, dim=0)
        else:
            aggregated = torch.mean(stacked, dim=0)
            
        return aggregated
    
    def aggregate_with_median(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate updates using median"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        stacked = torch.stack(updates)
        return torch.median(stacked, dim=0)[0]
    
    def filter_byzantine_updates(self, client_ids: List[str], 
                               updates: List[torch.Tensor]) -> tuple:
        """Filter out Byzantine updates"""
        byzantine_clients = self.detect_byzantine_clients(client_ids, updates)
        
        filtered_ids = []
        filtered_updates = []
        
        for i, (client_id, update) in enumerate(zip(client_ids, updates)):
            if client_id not in byzantine_clients:
                filtered_ids.append(client_id)
                filtered_updates.append(update)
        
        return filtered_ids, filtered_updates, byzantine_clients


# Security configuration constants
SECURITY_CONFIGS = {
    'differential_privacy': {
        'epsilon': 1.0,
        'delta': 1e-5,
        'sensitivity': 1.0,
        'noise_multiplier': 0.1
    },
    'byzantine_fault_tolerance': {
        'anomaly_threshold': 2.0,
        'min_clients': 3,
        'trim_ratio': 0.2
    }
}
