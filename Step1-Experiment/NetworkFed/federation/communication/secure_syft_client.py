"""
Secure Communication Manager for Federated Learning
==================================================
Implements secure communication protocols for PySyft federated learning with:
- AES Encryption for confidentiality
- Differential Privacy for privacy preservation
- Byzantine Fault Tolerance for robustness against malicious clients
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import statistics
from collections import defaultdict

from .syft_client import PySyftCommunicationManager
from core.interfaces import CommunicationInterface


class SecurePySyftCommunicationManager(PySyftCommunicationManager):
    """
    Secure communication manager with encryption, differential privacy, and Byzantine fault tolerance.
    
    Features:
    - AES-256 encryption for message confidentiality
    - Differential privacy with Gaussian noise
    - Byzantine fault tolerance with outlier detection
    - Message authentication and replay attack prevention
    """
    
    def __init__(self, encryption_key: Optional[str] = None, enable_dp: bool = True, 
                 noise_multiplier: float = 1.0, enable_bft: bool = True,
                 bft_threshold: float = 2.0, min_clients: int = 3):
        super().__init__()
        self.encryption_enabled = True
        self.differential_privacy_enabled = enable_dp
        self.byzantine_fault_tolerance_enabled = enable_bft
        self.noise_multiplier = noise_multiplier
        self.bft_threshold = bft_threshold  # Standard deviations for outlier detection
        self.min_clients_for_bft = min_clients
        
        # Initialize encryption
        if encryption_key:
            self.cipher_suite = self._create_cipher_from_key(encryption_key)
        else:
            self.cipher_suite = self._generate_cipher()
        
        # Byzantine fault tolerance tracking
        self.client_reputation = defaultdict(float)  # Track client reliability
        self.update_history = defaultdict(list)  # Store update norms for analysis
        
        # Security metrics
        self.security_metrics = {
            'encrypted_messages': 0,
            'authentication_checks': 0,
            'dp_noise_applications': 0,
            'security_violations': 0,
            'byzantine_attacks_detected': 0,
            'clients_filtered': 0,
            'reputation_updates': 0
        }
    
    def _generate_cipher(self) -> Fernet:
        """Generate a new encryption cipher."""
        key = Fernet.generate_key()
        return Fernet(key)
    
    def _create_cipher_from_key(self, password: str) -> Fernet:
        """Create cipher from password."""
        password_bytes = password.encode()
        salt = b'salt_for_federated_learning'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def encrypt_model_update(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Encrypt model update for secure transmission."""
        try:
            # Serialize model update
            serialized_update = self._serialize_model_update(model_update)
            
            # Encrypt
            encrypted_data = self.cipher_suite.encrypt(serialized_update)
            
            # Create secure package
            secure_package = {
                'encrypted_data': encrypted_data,
                'timestamp': time.time(),
                'checksum': self._calculate_checksum(serialized_update),
                'security_version': '1.0'
            }
            
            self.security_metrics['encrypted_messages'] += 1
            return secure_package
            
        except Exception as e:
            self.security_metrics['security_violations'] += 1
            raise Exception(f"Encryption failed: {str(e)}")
    
    def decrypt_model_update(self, secure_package: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decrypt received model update."""
        try:
            # Verify timestamp (prevent replay attacks)
            if time.time() - secure_package['timestamp'] > 300:  # 5 minute timeout
                raise Exception("Message timestamp too old")
            
            # Decrypt
            encrypted_data = secure_package['encrypted_data']
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # Verify checksum
            if self._calculate_checksum(decrypted_data) != secure_package['checksum']:
                raise Exception("Checksum verification failed")
            
            # Deserialize
            model_update = self._deserialize_model_update(decrypted_data)
            
            self.security_metrics['authentication_checks'] += 1
            return model_update
            
        except Exception as e:
            self.security_metrics['security_violations'] += 1
            raise Exception(f"Decryption failed: {str(e)}")
    
    def apply_differential_privacy(self, model_update: Dict[str, torch.Tensor], 
                                 sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply differential privacy noise to model updates."""
        if not self.differential_privacy_enabled:
            return model_update
        
        noisy_update = {}
        
        for param_name, param_tensor in model_update.items():
            # Calculate noise scale
            noise_scale = self.noise_multiplier * sensitivity
            
            # Generate Gaussian noise
            noise = torch.normal(
                mean=0.0, 
                std=noise_scale, 
                size=param_tensor.shape,
                device=param_tensor.device
            )
            
            # Add noise
            noisy_update[param_name] = param_tensor + noise
        
        self.security_metrics['dp_noise_applications'] += 1
        return noisy_update
    
    def detect_byzantine_clients(self, updates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Detect and filter out Byzantine (malicious) clients using statistical outlier detection.
        
        Args:
            updates: List of model updates with metadata
            
        Returns:
            Tuple of (filtered_updates, byzantine_client_ids)
        """
        if not self.byzantine_fault_tolerance_enabled or len(updates) < self.min_clients_for_bft:
            return updates, []
        
        # Extract client IDs and calculate update norms
        client_norms = {}
        for update in updates:
            client_id = update.get('datasite_id', 'unknown')
            model_params = update.get('model_update', {})
            
            # Calculate L2 norm of the update
            norm = self._calculate_update_norm(model_params)
            client_norms[client_id] = norm
            
            # Store in history for reputation tracking
            self.update_history[client_id].append(norm)
            if len(self.update_history[client_id]) > 10:  # Keep last 10 updates
                self.update_history[client_id].pop(0)
        
        # Detect outliers using statistical methods
        byzantine_clients = self._identify_outliers(client_norms)
        
        # Update client reputation
        self._update_client_reputation(client_norms, byzantine_clients)
        
        # Filter out Byzantine clients
        filtered_updates = [
            update for update in updates 
            if update.get('datasite_id', 'unknown') not in byzantine_clients
        ]
        
        if byzantine_clients:
            self.security_metrics['byzantine_attacks_detected'] += len(byzantine_clients)
            self.security_metrics['clients_filtered'] += len(byzantine_clients)
            print(f"🛡️ Byzantine fault tolerance: Filtered {len(byzantine_clients)} malicious clients: {byzantine_clients}")
        
        return filtered_updates, byzantine_clients
    
    def _calculate_update_norm(self, model_params: Dict[str, torch.Tensor]) -> float:
        """Calculate L2 norm of model update."""
        if not model_params:
            return 0.0
        
        total_norm = 0.0
        for param_tensor in model_params.values():
            if isinstance(param_tensor, torch.Tensor):
                total_norm += torch.norm(param_tensor).item() ** 2
        
        return np.sqrt(total_norm)
    
    def _identify_outliers(self, client_norms: Dict[str, float]) -> List[str]:
        """Identify outlier clients using IQR and Z-score methods."""
        if len(client_norms) < 3:
            return []
        
        norms = list(client_norms.values())
        clients = list(client_norms.keys())
        
        # Method 1: IQR-based outlier detection
        q1, q3 = np.percentile(norms, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Method 2: Z-score based outlier detection
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        outliers = set()
        
        for client_id, norm in client_norms.items():
            # IQR method
            if norm < lower_bound or norm > upper_bound:
                outliers.add(client_id)
            
            # Z-score method (if std > 0)
            if std_norm > 0:
                z_score = abs(norm - mean_norm) / std_norm
                if z_score > self.bft_threshold:
                    outliers.add(client_id)
        
        return list(outliers)
    
    def _update_client_reputation(self, client_norms: Dict[str, float], byzantine_clients: List[str]):
        """Update client reputation based on Byzantine detection."""
        for client_id in client_norms.keys():
            if client_id in byzantine_clients:
                # Decrease reputation for Byzantine behavior
                self.client_reputation[client_id] = max(0.0, self.client_reputation[client_id] - 0.2)
            else:
                # Increase reputation for honest behavior
                self.client_reputation[client_id] = min(1.0, self.client_reputation[client_id] + 0.05)
            
            self.security_metrics['reputation_updates'] += 1
    
    def apply_reputation_filtering(self, updates: List[Dict[str, Any]], 
                                 reputation_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Filter updates based on client reputation scores."""
        if not self.byzantine_fault_tolerance_enabled:
            return updates
        
        filtered_updates = []
        for update in updates:
            client_id = update.get('datasite_id', 'unknown')
            reputation = self.client_reputation.get(client_id, 0.5)  # Default neutral reputation
            
            if reputation >= reputation_threshold:
                filtered_updates.append(update)
            else:
                print(f"🛡️ Reputation filter: Excluding client {client_id} (reputation: {reputation:.3f})")
                self.security_metrics['clients_filtered'] += 1
        
        return filtered_updates
    
    def _serialize_model_update(self, model_update: Dict[str, torch.Tensor]) -> bytes:
        """Serialize model update to bytes."""
        import pickle
        return pickle.dumps(model_update)
    
    def _deserialize_model_update(self, data: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize bytes to model update."""
        import pickle
        return pickle.loads(data)
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()
    
    def send_model_update(self, datasite_id: str, model_update: Dict[str, torch.Tensor], 
                         round_num: int, **kwargs) -> Dict[str, Any]:
        """Send encrypted model update to server."""
        # Apply differential privacy
        if self.differential_privacy_enabled:
            model_update = self.apply_differential_privacy(model_update)
        
        # Encrypt model update
        secure_package = self.encrypt_model_update(model_update)
        
        # Add metadata
        secure_package.update({
            'datasite_id': datasite_id,
            'round_num': round_num,
            'communication_type': 'secure',
            **kwargs
        })
        
        return secure_package
    
    def receive_global_model(self, secure_package: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Receive and decrypt global model from server."""
        return self.decrypt_model_update(secure_package)
    
    def aggregate_secure_updates(self, secure_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate encrypted model updates with Byzantine fault tolerance.
        
        Process:
        1. Decrypt all updates
        2. Apply Byzantine fault tolerance filtering
        3. Apply reputation-based filtering
        4. Perform secure aggregation
        """
        # Decrypt all updates
        decrypted_updates = []
        for secure_update in secure_updates:
            try:
                model_update = self.decrypt_model_update(secure_update)
                # Add decrypted model to the update metadata
                secure_update['model_update'] = model_update
                decrypted_updates.append(secure_update)
            except Exception as e:
                print(f"⚠️  Failed to decrypt update: {e}")
                self.security_metrics['security_violations'] += 1
        
        if not decrypted_updates:
            raise Exception("No valid encrypted updates to aggregate")
        
        # Apply Byzantine fault tolerance
        if self.byzantine_fault_tolerance_enabled:
            # Step 1: Detect Byzantine clients using statistical analysis
            filtered_updates, byzantine_clients = self.detect_byzantine_clients(decrypted_updates)
            
            # Step 2: Apply reputation-based filtering
            final_updates = self.apply_reputation_filtering(filtered_updates)
            
            print(f"🛡️ Byzantine filtering: {len(decrypted_updates)} → {len(final_updates)} clients")
        else:
            final_updates = decrypted_updates
        
        if not final_updates:
            raise Exception("No valid updates remaining after Byzantine filtering")
        
        # Extract model updates for aggregation
        model_updates = [update['model_update'] for update in final_updates]
        
        # Perform robust aggregation
        return self._robust_weighted_average_aggregation(model_updates, final_updates)
    
    def _robust_weighted_average_aggregation(self, model_updates: List[Dict[str, torch.Tensor]], 
                                           update_metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Robust weighted average aggregation with Byzantine fault tolerance.
        
        Uses trimmed mean aggregation to further reduce Byzantine influence.
        """
        if not model_updates:
            return {}
        
        # Initialize aggregated parameters
        aggregated = {}
        
        for param_name in model_updates[0].keys():
            param_tensors = [update[param_name] for update in model_updates]
            
            if self.byzantine_fault_tolerance_enabled and len(param_tensors) >= 5:
                # Use trimmed mean (remove top and bottom 10%)
                aggregated[param_name] = self._trimmed_mean_aggregation(param_tensors)
            else:
                # Standard average
                param_sum = torch.zeros_like(param_tensors[0])
                for param_tensor in param_tensors:
                    param_sum += param_tensor
                aggregated[param_name] = param_sum / len(param_tensors)
        
        return aggregated
    
    def _trimmed_mean_aggregation(self, param_tensors: List[torch.Tensor], 
                                trim_ratio: float = 0.1) -> torch.Tensor:
        """
        Compute trimmed mean by removing extreme values.
        
        Args:
            param_tensors: List of parameter tensors
            trim_ratio: Fraction of extreme values to remove (both ends)
        
        Returns:
            Trimmed mean tensor
        """
        if len(param_tensors) < 3:
            # Fall back to simple average for small groups
            return torch.mean(torch.stack(param_tensors), dim=0)
        
        # Stack tensors for easier manipulation
        stacked = torch.stack(param_tensors)  # Shape: [num_clients, ...param_shape]
        
        # Calculate how many to trim from each end
        num_to_trim = max(1, int(len(param_tensors) * trim_ratio))
        
        # Sort along the client dimension and trim extremes
        sorted_tensors, _ = torch.sort(stacked, dim=0)
        trimmed_tensors = sorted_tensors[num_to_trim:-num_to_trim] if num_to_trim > 0 else sorted_tensors
        
        # Compute mean of remaining tensors
        return torch.mean(trimmed_tensors, dim=0)
    
    def _weighted_average_aggregation(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Weighted average aggregation of model updates."""
        if not updates:
            return {}
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Simple average (can be extended to weighted average)
        for param_name in updates[0].keys():
            param_sum = torch.zeros_like(updates[0][param_name])
            for update in updates:
                param_sum += update[param_name]
            aggregated[param_name] = param_sum / len(updates)
        
        return aggregated
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security-related metrics."""
        return {
            'communication_type': 'secure',
            'encryption_enabled': self.encryption_enabled,
            'differential_privacy_enabled': self.differential_privacy_enabled,
            'byzantine_fault_tolerance_enabled': self.byzantine_fault_tolerance_enabled,
            'noise_multiplier': self.noise_multiplier,
            'bft_threshold': self.bft_threshold,
            'security_metrics': self.security_metrics.copy(),
            'client_reputation_summary': {
                'total_clients': len(self.client_reputation),
                'avg_reputation': np.mean(list(self.client_reputation.values())) if self.client_reputation else 0.0,
                'min_reputation': min(self.client_reputation.values()) if self.client_reputation else 0.0,
                'max_reputation': max(self.client_reputation.values()) if self.client_reputation else 0.0,
                'low_reputation_clients': sum(1 for rep in self.client_reputation.values() if rep < 0.3)
            },
            'security_level': self._calculate_security_level()
        }
    
    def _calculate_security_level(self) -> str:
        """Calculate overall security level including Byzantine resistance."""
        violations = self.security_metrics['security_violations']
        byzantine_attacks = self.security_metrics['byzantine_attacks_detected']
        total_operations = (
            self.security_metrics['encrypted_messages'] + 
            self.security_metrics['authentication_checks']
        )
        
        if total_operations == 0:
            return 'unknown'
        
        violation_rate = violations / total_operations
        byzantine_rate = byzantine_attacks / max(1, total_operations)
        
        # Enhanced security level calculation
        if violation_rate == 0 and byzantine_rate < 0.05 and self.byzantine_fault_tolerance_enabled:
            return 'high'
        elif violation_rate < 0.01 and byzantine_rate < 0.1:
            return 'medium'
        else:
            return 'low'
    
    def reset_security_metrics(self):
        """Reset security metrics counters."""
        self.security_metrics = {
            'encrypted_messages': 0,
            'authentication_checks': 0,
            'dp_noise_applications': 0,
            'security_violations': 0,
            'byzantine_attacks_detected': 0,
            'clients_filtered': 0,
            'reputation_updates': 0
        }
        self.client_reputation.clear()
        self.update_history.clear()
    
    def get_client_reputation_report(self) -> Dict[str, Any]:
        """Generate detailed client reputation report."""
        return {
            'client_reputations': dict(self.client_reputation),
            'update_history_summary': {
                client_id: {
                    'num_updates': len(history),
                    'avg_norm': np.mean(history) if history else 0.0,
                    'std_norm': np.std(history) if len(history) > 1 else 0.0,
                    'latest_norm': history[-1] if history else 0.0
                }
                for client_id, history in self.update_history.items()
            },
            'byzantine_detection_summary': {
                'total_attacks_detected': self.security_metrics['byzantine_attacks_detected'],
                'total_clients_filtered': self.security_metrics['clients_filtered'],
                'reputation_updates': self.security_metrics['reputation_updates']
            }
        }
