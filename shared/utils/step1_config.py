#!/usr/bin/env python3
"""
Configuration Management for Federated Learning Predictive Maintenance
Centralized configuration management for experiments and model training
"""

import os
import json
import yaml
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architectures"""
    # CNN Configuration
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 3
    cnn_activation: str = 'relu'
    cnn_dropout: float = 0.3
    cnn_batch_norm: bool = True
    
    # LSTM Configuration  
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2
    lstm_bidirectional: bool = True
    lstm_return_sequences: bool = False
    
    # Hybrid Model Configuration
    hybrid_cnn_filters: List[int] = field(default_factory=lambda: [32, 64])
    hybrid_lstm_units: int = 64
    hybrid_fusion_units: List[int] = field(default_factory=lambda: [128, 64])
    
    # Common model parameters
    dense_units: List[int] = field(default_factory=lambda: [128, 64])
    output_activation: str = 'softmax'
    use_attention: bool = False
    attention_units: int = 64

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
    loss_function: str = 'sparse_categorical_crossentropy'
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall'])
    
    # Regularization
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_loss'
    reduce_lr_on_plateau: bool = True
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Class weights
    use_class_weights: bool = True
    class_weight_method: str = 'balanced'  # 'balanced', 'custom'
    
    # Validation
    validation_split: float = 0.0  # We use separate validation set
    shuffle: bool = True
    
    # Checkpointing
    save_best_only: bool = True
    save_weights_only: bool = False
    monitor_metric: str = 'val_accuracy'

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    # Client configuration
    num_clients: int = 10
    clients_per_round: int = 5  # Number of clients to sample per round
    min_clients: int = 3  # Minimum clients needed for training
    
    # Communication rounds
    num_rounds: int = 50
    local_epochs: int = 5
    
    # Aggregation strategy
    aggregation_method: str = 'fedavg'  # 'fedavg', 'fedprox', 'fedopt'
    aggregation_weights: str = 'uniform'  # 'uniform', 'data_size'
    
    # FedProx specific parameters
    fedprox_mu: float = 0.01  # Proximal regularization strength
    
    # Privacy parameters
    use_differential_privacy: bool = False
    dp_noise_multiplier: float = 1.1
    dp_l2_norm_clip: float = 1.0
    
    # Communication efficiency
    compression_method: str = 'none'  # 'none', 'quantization', 'sparsification'
    quantization_bits: int = 8
    sparsification_ratio: float = 0.1
    
    # Client sampling
    sampling_strategy: str = 'random'  # 'random', 'round_robin', 'performance_based'
    
    # Fault tolerance
    max_client_failures: int = 2
    client_timeout: float = 300.0  # seconds

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Experiment identification
    experiment_name: str = "federated_predictive_maintenance"
    experiment_description: str = "Federated learning for industrial predictive maintenance"
    experiment_version: str = "1.0"
    
    # Paths
    base_dir: str = "/workspaces/pdm-fdl/Step1-Experiment"
    data_dir: str = "/workspaces/pdm-fdl/shared/processed_data"
    results_dir: str = "/workspaces/pdm-fdl/Step1-Experiment/results"
    models_dir: str = "/workspaces/pdm-fdl/Step1-Experiment/models"
    logs_dir: str = "/workspaces/pdm-fdl/Step1-Experiment/logs"
    
    # Data configuration
    data_format: str = 'both'  # 'tabular', 'sequences', 'both'
    target_type: str = 'binary'  # 'binary', 'multiclass', 'both'
    
    # Model types to train
    model_types: List[str] = field(default_factory=lambda: ['cnn', 'lstm', 'hybrid'])
    
    # Evaluation configuration
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'confusion_matrix'
    ])
    
    # Comparison baselines
    include_centralized_baseline: bool = True
    include_local_baseline: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_time_hours: float = 24.0
    
    # Logging
    log_level: str = 'INFO'
    log_frequency: int = 10  # Log every N rounds
    save_intermediate_results: bool = True

@dataclass
class SystemConfig:
    """Configuration for system resources and environment"""
    # Compute resources
    use_gpu: bool = False
    gpu_memory_limit: Optional[float] = None
    cpu_threads: int = -1  # -1 for all available
    
    # Distributed computing
    use_distributed: bool = False
    master_addr: str = 'localhost'
    master_port: int = 12355
    
    # Memory management
    memory_efficient: bool = True
    gradient_accumulation_steps: int = 1
    
    # Debugging
    debug_mode: bool = False
    profile_performance: bool = False
    
    # Security
    secure_communication: bool = True
    encryption_key_path: Optional[str] = None

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.federated_config = FederatedConfig()
        self.experiment_config = ExperimentConfig()
        self.system_config = SystemConfig()
        
        # Load config if path provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file"""
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Update configurations
        if 'model' in config_dict:
            self._update_dataclass(self.model_config, config_dict['model'])
        
        if 'training' in config_dict:
            self._update_dataclass(self.training_config, config_dict['training'])
        
        if 'federated' in config_dict:
            self._update_dataclass(self.federated_config, config_dict['federated'])
        
        if 'experiment' in config_dict:
            self._update_dataclass(self.experiment_config, config_dict['experiment'])
        
        if 'system' in config_dict:
            self._update_dataclass(self.system_config, config_dict['system'])
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file"""
        logger.info(f"Saving configuration to {config_path}")
        
        config_dict = {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'federated': asdict(self.federated_config),
            'experiment': asdict(self.experiment_config),
            'system': asdict(self.system_config)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary values"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as dictionary"""
        return {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'federated': asdict(self.federated_config),
            'experiment': asdict(self.experiment_config),
            'system': asdict(self.system_config)
        }
    
    def validate_config(self) -> bool:
        """Validate configuration consistency"""
        issues = []
        
        # Check path validity
        required_dirs = [
            self.experiment_config.base_dir,
            self.experiment_config.data_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"Required directory does not exist: {dir_path}")
        
        # Check federated learning parameters
        if self.federated_config.clients_per_round > self.federated_config.num_clients:
            issues.append("clients_per_round cannot be greater than num_clients")
        
        if self.federated_config.min_clients > self.federated_config.clients_per_round:
            issues.append("min_clients cannot be greater than clients_per_round")
        
        # Check training parameters
        if self.training_config.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        if self.training_config.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        
        # Check data format and model compatibility
        if 'lstm' in self.experiment_config.model_types:
            if self.experiment_config.data_format not in ['sequences', 'both']:
                issues.append("LSTM models require sequence data format")
        
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def setup_experiment_directories(self) -> None:
        """Create experiment directory structure"""
        directories = [
            self.experiment_config.results_dir,
            self.experiment_config.models_dir,
            self.experiment_config.logs_dir,
            os.path.join(self.experiment_config.results_dir, "plots"),
            os.path.join(self.experiment_config.results_dir, "metrics"),
            os.path.join(self.experiment_config.models_dir, "centralized"),
            os.path.join(self.experiment_config.models_dir, "federated"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def get_model_config_for_type(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        base_config = {
            'dense_units': self.model_config.dense_units,
            'output_activation': self.model_config.output_activation,
            'use_attention': self.model_config.use_attention,
            'attention_units': self.model_config.attention_units
        }
        
        if model_type.lower() == 'cnn':
            base_config.update({
                'filters': self.model_config.cnn_filters,
                'kernel_size': self.model_config.cnn_kernel_size,
                'activation': self.model_config.cnn_activation,
                'dropout': self.model_config.cnn_dropout,
                'batch_norm': self.model_config.cnn_batch_norm
            })
        
        elif model_type.lower() == 'lstm':
            base_config.update({
                'units': self.model_config.lstm_units,
                'dropout': self.model_config.lstm_dropout,
                'recurrent_dropout': self.model_config.lstm_recurrent_dropout,
                'bidirectional': self.model_config.lstm_bidirectional,
                'return_sequences': self.model_config.lstm_return_sequences
            })
        
        elif model_type.lower() == 'hybrid':
            base_config.update({
                'cnn_filters': self.model_config.hybrid_cnn_filters,
                'lstm_units': self.model_config.hybrid_lstm_units,
                'fusion_units': self.model_config.hybrid_fusion_units
            })
        
        return base_config
    
    def create_experiment_summary(self) -> Dict[str, Any]:
        """Create experiment summary for documentation"""
        return {
            'experiment_info': {
                'name': self.experiment_config.experiment_name,
                'description': self.experiment_config.experiment_description,
                'version': self.experiment_config.experiment_version,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'model_types': self.experiment_config.model_types,
            'federated_setup': {
                'num_clients': self.federated_config.num_clients,
                'num_rounds': self.federated_config.num_rounds,
                'local_epochs': self.federated_config.local_epochs,
                'aggregation_method': self.federated_config.aggregation_method
            },
            'training_setup': {
                'epochs': self.training_config.epochs,
                'batch_size': self.training_config.batch_size,
                'learning_rate': self.training_config.learning_rate,
                'optimizer': self.training_config.optimizer
            },
            'evaluation_metrics': self.experiment_config.evaluation_metrics,
            'random_seed': self.experiment_config.random_seed
        }

def create_default_config(config_path: str) -> ConfigManager:
    """Create default configuration file"""
    config_manager = ConfigManager()
    config_manager.save_config(config_path)
    logger.info(f"Created default configuration at {config_path}")
    return config_manager

def load_config_from_file(config_path: str) -> ConfigManager:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}. Creating default config.")
        return create_default_config(config_path)
    
    return ConfigManager(config_path)

# Example configuration templates
DEFAULT_CONFIG_TEMPLATE = {
    "experiment": {
        "experiment_name": "federated_predictive_maintenance_v1",
        "experiment_description": "Federated learning for industrial predictive maintenance using AI4I 2020 dataset",
        "model_types": ["cnn", "lstm", "hybrid"],
        "include_centralized_baseline": True
    },
    "federated": {
        "num_clients": 10,
        "num_rounds": 50,
        "local_epochs": 5,
        "aggregation_method": "fedavg",
        "clients_per_round": 8
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "early_stopping": True,
        "early_stopping_patience": 10
    },
    "model": {
        "cnn_filters": [32, 64, 128],
        "lstm_units": [64, 32],
        "cnn_dropout": 0.3,
        "lstm_dropout": 0.2
    }
}

if __name__ == "__main__":
    # Example usage
    config_path = "/workspaces/pdm-fdl/shared/utils/step1_config.yaml"
    
    # Create default config
    config_manager = create_default_config(config_path)
    
    # Validate and setup
    if config_manager.validate_config():
        config_manager.setup_experiment_directories()
        print("Configuration setup completed successfully!")
    else:
        print("Configuration validation failed!")
