"""
Secure Federated Server Implementation
Extends the base FederatedServer with security features
"""
import torch
from typing import List, Dict, Any, Optional, Set
from federated_server import FederatedServer
from security_managers import DifferentialPrivacyManager, ByzantineFaultTolerance, SECURITY_CONFIGS


class SecureFederatedServer(FederatedServer):
    """
    Secure Federated Server with Differential Privacy and Byzantine Fault Tolerance
    """
    
    def __init__(self, model_class, model_params: Dict[str, Any], 
                 aggregation_method: str = 'fedavg', device: str = 'cpu',
                 privacy_enabled: bool = False, early_stopping_patience: int = 5,
                 early_stopping_min_delta: float = 0.001,
                 enable_dp: bool = False, enable_bft: bool = False,
                 security_configs: Optional[Dict] = None):
        """
        Initialize Secure Federated Server
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            aggregation_method: Method for aggregating client updates
            device: Computing device (CPU/GPU)
            privacy_enabled: Enable privacy features
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum delta for early stopping
            enable_dp: Enable differential privacy
            enable_bft: Enable Byzantine fault tolerance
            security_configs: Security configuration parameters
        """
        # Initialize parent class
        super().__init__(
            model_class=model_class,
            model_params=model_params,
            aggregation_method=aggregation_method,
            device=device,
            privacy_enabled=privacy_enabled,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        
        self.enable_dp = enable_dp
        self.enable_bft = enable_bft
        
        # Security managers
        if security_configs is None:
            security_configs = SECURITY_CONFIGS
            
        if enable_dp:
            dp_config = security_configs['differential_privacy']
            self.dp_manager = DifferentialPrivacyManager(
                epsilon=dp_config['epsilon'],
                delta=dp_config['delta'],
                sensitivity=dp_config['sensitivity'],
                noise_multiplier=dp_config['noise_multiplier']
            )
        else:
            self.dp_manager = None
            
        if enable_bft:
            bft_config = security_configs['byzantine_fault_tolerance']
            self.bft_manager = ByzantineFaultTolerance(
                anomaly_threshold=bft_config['anomaly_threshold'],
                min_clients=bft_config['min_clients']
            )
        else:
            self.bft_manager = None
    
    def aggregate_updates(self, client_updates: List[Dict], 
                         algorithm_kwargs: Dict = None):
        """
        Aggregate client updates with security features
        
        Args:
            client_updates: List of client update dictionaries
            algorithm_kwargs: Algorithm-specific parameters
        """
        if algorithm_kwargs is None:
            algorithm_kwargs = {}
            
        if not client_updates:
            return super().aggregate_updates(client_updates, algorithm_kwargs)
        
        # Extract client IDs and model updates
        client_ids = [update.get('client_id', f'client_{i}') for i, update in enumerate(client_updates)]
        model_updates = [update.get('model_update', {}) for update in client_updates]
        
        if not any(model_updates):
            return super().aggregate_updates(client_updates, algorithm_kwargs)
        
        # Convert updates to tensors for security analysis
        update_tensors = []
        for update in model_updates:
            if update:
                # Flatten all parameters into a single tensor
                flattened = torch.cat([param.flatten() for param in update.values()])
                update_tensors.append(flattened)
        
        if not update_tensors:
            return super().aggregate_updates(client_updates, algorithm_kwargs)
        
        # Apply Byzantine fault tolerance if enabled
        if self.enable_bft and self.bft_manager:
            filtered_ids, filtered_updates, byzantine_clients = self.bft_manager.filter_byzantine_updates(
                client_ids, update_tensors
            )
            
            if byzantine_clients:
                print(f"Detected Byzantine clients: {byzantine_clients}")
            
            # Use filtered updates for aggregation
            if filtered_updates:
                # Convert back to original format for aggregation
                filtered_client_updates = []
                param_shapes = {}
                param_names = list(model_updates[0].keys()) if model_updates[0] else []
                
                # Get parameter shapes
                if param_names:
                    for name in param_names:
                        param_shapes[name] = model_updates[0][name].shape
                        param_shapes[f"{name}_size"] = model_updates[0][name].numel()
                    
                    # Reconstruct filtered updates
                    for i, filtered_tensor in enumerate(filtered_updates):
                        reconstructed_update = {}
                        start_idx = 0
                        for name in param_names:
                            size = param_shapes[f"{name}_size"]
                            shape = param_shapes[name]
                            reconstructed_update[name] = filtered_tensor[start_idx:start_idx+size].reshape(shape)
                            start_idx += size
                        
                        # Create new client update dict
                        new_client_update = client_updates[client_ids.index(filtered_ids[i])].copy()
                        new_client_update['model_update'] = reconstructed_update
                        filtered_client_updates.append(new_client_update)
                    
                    client_updates = filtered_client_updates
                else:
                    print("Warning: No model updates found, using original updates")
            else:
                print("Warning: All clients filtered as Byzantine, using original updates")
        
        # Apply differential privacy if enabled
        if self.enable_dp and self.dp_manager:
            # Add noise to each client update
            noisy_client_updates = []
            for update in client_updates:
                model_update = update.get('model_update', {})
                if model_update:
                    noisy_update = {}
                    for param_name, param_tensor in model_update.items():
                        noisy_param = self.dp_manager.add_noise_to_model_update(param_tensor)
                        noisy_update[param_name] = noisy_param
                    
                    new_client_update = update.copy()
                    new_client_update['model_update'] = noisy_update
                    noisy_client_updates.append(new_client_update)
                else:
                    noisy_client_updates.append(update)
            client_updates = noisy_client_updates
        
        # Use parent class aggregation method
        return super().aggregate_updates(client_updates, algorithm_kwargs)
    
    def get_security_metrics(self, round_num: int) -> Dict[str, Any]:
        """Get security-related metrics"""
        metrics = {}
        
        if self.enable_dp and self.dp_manager:
            epsilon_spent, delta_spent = self.dp_manager.get_privacy_spent(round_num)
            metrics['differential_privacy'] = {
                'epsilon_spent': epsilon_spent,
                'delta_spent': delta_spent,
                'privacy_budget_remaining': max(0, 10.0 - epsilon_spent)  # Assuming budget of 10
            }
        
        if self.enable_bft and self.bft_manager:
            metrics['byzantine_fault_tolerance'] = {
                'anomaly_threshold': self.bft_manager.anomaly_threshold,
                'min_clients': self.bft_manager.min_clients
            }
        
        return metrics
    
    def update_security_config(self, config_updates: Dict[str, Any]):
        """Update security configuration"""
        if 'differential_privacy' in config_updates and self.dp_manager:
            dp_config = config_updates['differential_privacy']
            if 'epsilon' in dp_config:
                self.dp_manager.epsilon = dp_config['epsilon']
            if 'noise_multiplier' in dp_config:
                self.dp_manager.noise_multiplier = dp_config['noise_multiplier']
        
        if 'byzantine_fault_tolerance' in config_updates and self.bft_manager:
            bft_config = config_updates['byzantine_fault_tolerance']
            if 'anomaly_threshold' in bft_config:
                self.bft_manager.anomaly_threshold = bft_config['anomaly_threshold']
