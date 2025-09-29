"""
Experiment Configuration Management
==================================
Handles parsing and validation of experiment configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import json

from core.enums import FederatedAlgorithm, ModelType, DataDistribution
from core.exceptions import ConfigurationError


@dataclass
class DataSiteConfig:
    """Configuration for a factory datasite."""
    datasite_id: str
    url: str
    credentials: Dict[str, str]
    region: str = "unknown"
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Basic experiment info
    experiment_id: str
    name: str
    description: str = ""
    
    # Model configuration
    model_type: ModelType = ModelType.CNN
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Algorithm configuration
    algorithm: FederatedAlgorithm = FederatedAlgorithm.FEDAVG
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    max_rounds: int = 10
    training_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data_distribution: DataDistribution = DataDistribution.IID
    data_params: Dict[str, Any] = field(default_factory=dict)
    
    # Datasite configuration
    datasites: List[DataSiteConfig] = field(default_factory=list)
    
    # Early stopping configuration
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Privacy configuration
    privacy_enabled: bool = False
    privacy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring configuration
    monitoring_enabled: bool = True
    monitoring_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary."""
        try:
            # Convert enums from strings
            if 'model_type' in config_dict:
                config_dict['model_type'] = ModelType(config_dict['model_type'])
            
            if 'algorithm' in config_dict:
                config_dict['algorithm'] = FederatedAlgorithm(config_dict['algorithm'])
            
            if 'data_distribution' in config_dict:
                config_dict['data_distribution'] = DataDistribution(config_dict['data_distribution'])
            
            # Convert datasites to DataSiteConfig objects
            if 'datasites' in config_dict:
                datasites = []
                for ds_config in config_dict['datasites']:
                    if isinstance(ds_config, dict):
                        datasites.append(DataSiteConfig(**ds_config))
                    else:
                        datasites.append(ds_config)
                config_dict['datasites'] = datasites
            
            return cls(**config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {str(e)}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML config: {str(e)}")
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load JSON config: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (ModelType, FederatedAlgorithm, DataDistribution)):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], DataSiteConfig):
                result[key] = [ds.__dict__ for ds in value]
            else:
                result[key] = value
        return result
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def save_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Basic validation
        if not self.experiment_id:
            errors.append("experiment_id is required")
        
        if not self.name:
            errors.append("name is required")
        
        if self.max_rounds <= 0:
            errors.append("max_rounds must be positive")
        
        # Datasite validation
        if not self.datasites:
            errors.append("At least one datasite must be configured")
        
        # Check for duplicate datasite IDs
        datasite_ids = [ds.datasite_id for ds in self.datasites]
        if len(datasite_ids) != len(set(datasite_ids)):
            errors.append("Duplicate datasite IDs found")
        
        # Algorithm-specific validation
        if self.algorithm == FederatedAlgorithm.FEDPROX:
            if 'mu' not in self.algorithm_params:
                errors.append("FedProx requires 'mu' parameter")
        
        return errors


class ConfigurationManager:
    """Manages experiment configurations and templates."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Template configurations
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load configuration templates."""
        # Basic CNN experiment template
        self.templates['basic_cnn'] = {
            'model_type': 'cnn',
            'algorithm': 'fedavg',
            'max_rounds': 100,  # Allow flexible rounds, will be overridden by command-line
            'model_params': {
                'input_dim': 10,
                'num_classes': 2
            },
            'training_params': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 1
            }
        }
        
        # FedProx experiment template
        self.templates['fedprox_lstm'] = {
            'model_type': 'lstm',
            'algorithm': 'fedprox',
            'max_rounds': 100,  # Allow flexible rounds, will be overridden by command-line
            'algorithm_params': {
                'mu': 0.01
            },
            'model_params': {
                'input_dim': 10,
                'num_classes': 2
            },
            'training_params': {
                'learning_rate': 0.0005,
                'batch_size': 16,
                'epochs': 1
            }
        }
        
        # Full experiment template
        self.templates['full_experiment'] = {
            'model_type': 'hybrid',
            'algorithm': 'fedavg',
            'max_rounds': 100,  # Allow flexible rounds, will be overridden by command-line
            'data_distribution': 'non_iid_label',
            'early_stopping_enabled': True,
            'early_stopping_patience': 5,
            'privacy_enabled': True,
            'monitoring_enabled': True
        }
    
    def create_from_template(self, template_name: str, experiment_id: str,
                           overrides: Dict[str, Any] = None) -> ExperimentConfig:
        """Create configuration from template."""
        if template_name not in self.templates:
            raise ConfigurationError(f"Template '{template_name}' not found")
        
        # Start with template
        config_dict = self.templates[template_name].copy()
        
        # Add required fields
        config_dict['experiment_id'] = experiment_id
        config_dict['name'] = f"Experiment {experiment_id}"
        
        # Apply overrides
        if overrides:
            config_dict.update(overrides)
        
        return ExperimentConfig.from_dict(config_dict)
    
    def save_template(self, template_name: str, template_config: Dict[str, Any]) -> None:
        """Save a new template."""
        self.templates[template_name] = template_config
        
        # Save to file
        template_path = self.config_dir / f"{template_name}_template.yaml"
        with open(template_path, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False)
    
    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())
    
    def generate_batch_configs(self, base_config: ExperimentConfig,
                             variations: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """
        Generate batch of configurations with parameter variations.
        
        Args:
            base_config: Base configuration to vary
            variations: Dict of parameter names to lists of values
            
        Returns:
            List of configuration variants
        """
        import itertools
        
        configs = []
        
        # Get all combinations of variations
        keys = list(variations.keys())
        values = list(variations.values())
        
        for i, combination in enumerate(itertools.product(*values)):
            # Create new config based on base
            config_dict = base_config.to_dict()
            
            # Apply variations
            for key, value in zip(keys, combination):
                if '.' in key:
                    # Handle nested parameters like 'algorithm_params.mu'
                    parts = key.split('.')
                    target = config_dict
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    config_dict[key] = value
            
            # Update experiment ID
            config_dict['experiment_id'] = f"{base_config.experiment_id}_var{i}"
            config_dict['name'] = f"{base_config.name} - Variation {i}"
            
            configs.append(ExperimentConfig.from_dict(config_dict))
        
        return configs
    
    def validate_config_file(self, config_path: str) -> List[str]:
        """Validate a configuration file and return errors."""
        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = ExperimentConfig.from_yaml(config_path)
            elif config_path.endswith('.json'):
                config = ExperimentConfig.from_json(config_path)
            else:
                return ["Unsupported file format. Use .yaml or .json"]
            
            return config.validate()
            
        except Exception as e:
            return [f"Failed to load configuration: {str(e)}"]
