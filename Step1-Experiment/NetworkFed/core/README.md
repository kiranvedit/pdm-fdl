# Federated Learning Core Framework

This directory contains the foundational core components that define the architecture and contracts for the federated learning system. It provides the essential building blocks through interfaces, enumerations, and exception handling that ensure consistency and maintainability across the entire framework.

## Architecture Overview

The core framework follows a clean architecture pattern with well-defined contracts:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Interfaces    │    │   Enumerations  │    │   Exceptions    │
│   - Contracts   │    │   - Constants   │    │   - Error       │
│   - Protocols   │    │   - Types       │    │     Handling    │
│   - Standards   │    │   - States      │    │   - Hierarchy   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Core Module    │
                    │  - Unified API  │
                    │  - Type Safety  │
                    │  - Consistency  │
                    └─────────────────┘
```

---

## 1. Core Interfaces (`interfaces.py`)

### What are Core Interfaces?
The interfaces module defines abstract base classes that establish contracts for all major components in the federated learning system. These interfaces ensure consistency, enable polymorphism, and provide clear architectural boundaries between different system components.

### Interface Architecture

#### **1. Model Interface (`ModelInterface`)**
```python
class ModelInterface(ABC):
    @abstractmethod
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return model parameters as a dictionary."""
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from a dictionary."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
```

**Purpose:**
- Standardizes model parameter access across all model types (CNN, LSTM, Hybrid)
- Enables seamless model parameter exchange in federated learning
- Provides consistent interface for model evaluation and inference

**Implementation Examples:**
- `OptimizedCNNModel`, `OptimizedLSTMModel`, `OptimizedHybridModel`
- Custom industrial IoT prediction models
- Third-party model adaptations

#### **2. Federated Algorithm Interface (`FederatedAlgorithmInterface`)**
```python
class FederatedAlgorithmInterface(ABC):
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates into global model parameters."""
    
    @abstractmethod
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure algorithm-specific parameters for client training."""
    
    @abstractmethod
    def update_algorithm_state(self, client_updates: List[Dict[str, Any]]) -> None:
        """Update algorithm-specific state after aggregation."""
```

**Purpose:**
- Defines contract for federated learning algorithms (FedAvg, FedProx, FedDyn, FedNova)
- Enables pluggable algorithm implementations
- Standardizes aggregation and state management

**Implementation Examples:**
- `FedAvgAlgorithm` - Standard federated averaging
- `FedProxAlgorithm` - Proximal term for heterogeneous clients
- `FedDynAlgorithm` - Dynamic regularization
- `FedNovaAlgorithm` - Normalized averaging

#### **3. DataSite Interface (`DataSiteInterface`)**
```python
class DataSiteInterface(ABC):
    @abstractmethod
    def initialize_datasite(self, config: Dict[str, Any]) -> None:
        """Initialize the PySyft datasite with configuration."""
    
    @abstractmethod
    def register_data(self, data: Dict[str, Any]) -> None:
        """Register local data as PySyft assets."""
    
    @abstractmethod
    def train_local_model(self, global_model: ModelInterface, 
                         training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train local model and return updates."""
    
    @abstractmethod
    def get_datasite_info(self) -> Dict[str, Any]:
        """Return datasite information and capabilities."""
```

**Purpose:**
- Standardizes industrial datasite operations
- Defines contract for local training on factory equipment
- Enables heterogeneous industrial site integration

**Industrial Context:**
- Each factory/manufacturing site implements this interface
- Handles local sensor data processing
- Manages equipment-specific model training

#### **4. Orchestrator Interface (`OrchestratorInterface`)**
```python
class OrchestratorInterface(ABC):
    @abstractmethod
    def setup_experiment(self, config: Dict[str, Any]) -> None:
        """Setup experiment with configuration."""
    
    @abstractmethod
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run a single federated learning experiment."""
    
    @abstractmethod
    def collect_results(self, experiment_id: str) -> Dict[str, Any]:
        """Collect and aggregate experiment results."""
```

**Purpose:**
- Defines experiment coordination and management
- Standardizes multi-site experiment execution
- Enables systematic research and production deployment

#### **5. Metrics Collector Interface (`MetricsCollectorInterface`)**
```python
class MetricsCollectorInterface(ABC):
    @abstractmethod
    def collect_round_metrics(self, round_num: int, 
                            global_model: ModelInterface,
                            client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect metrics for a single round."""
    
    @abstractmethod
    def collect_experiment_metrics(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect overall experiment metrics."""
    
    @abstractmethod
    def export_metrics(self, filepath: str) -> None:
        """Export collected metrics to file."""
```

**Purpose:**
- Standardizes performance monitoring and analysis
- Enables consistent metrics collection across experiments
- Provides foundation for research analysis and production monitoring

#### **6. Communication Interface (`CommunicationInterface`)**
```python
class CommunicationInterface(ABC):
    @abstractmethod
    def connect_to_datasite(self, datasite_url: str, credentials: Dict[str, str]) -> bool:
        """Connect to a factory datasite."""
    
    @abstractmethod
    def send_model(self, model_parameters: Dict[str, torch.Tensor], 
                   datasite_id: str) -> bool:
        """Send model parameters to a datasite."""
    
    @abstractmethod
    def request_training(self, training_config: Dict[str, Any], 
                        datasite_id: str) -> Dict[str, Any]:
        """Request local training from a datasite."""
    
    @abstractmethod
    def disconnect_from_datasite(self, datasite_id: str) -> bool:
        """Disconnect from a factory datasite."""
```

**Purpose:**
- Abstracts PySyft communication protocols
- Enables different communication backend implementations
- Standardizes distributed system interactions

### Why These Interfaces?

#### **Design Principles**
- **Separation of Concerns**: Each interface has a single, well-defined responsibility
- **Dependency Inversion**: High-level modules depend on abstractions, not concretions
- **Open/Closed Principle**: Open for extension, closed for modification
- **Interface Segregation**: Clients depend only on methods they use

#### **Industrial Benefits**
- **Modularity**: Easy to swap implementations for different industrial environments
- **Testability**: Mock implementations for testing without real industrial equipment
- **Scalability**: Add new algorithms, models, or communication protocols without breaking existing code
- **Maintainability**: Clear contracts reduce coupling and improve code quality

---

## 2. System Enumerations (`enums.py`)

### What are System Enumerations?
Enumerations provide type-safe constants that represent the various states, types, and configurations available throughout the federated learning system. They ensure consistency and prevent invalid value assignments.

### Enumeration Categories

#### **1. Federated Algorithm Types (`FederatedAlgorithm`)**
```python
class FederatedAlgorithm(Enum):
    FEDAVG = "fedavg"      # Standard federated averaging
    FEDPROX = "fedprox"    # Proximal term for heterogeneous clients
    FEDDYN = "feddyn"      # Dynamic regularization
    FEDNOVA = "fednova"    # Normalized averaging
```

**Usage:**
```python
from core import FederatedAlgorithm

# Type-safe algorithm selection
algorithm = FederatedAlgorithm.FEDAVG
server = FederatedServer(aggregation_method=algorithm)
```

**Why These Algorithms:**
- **FedAvg**: Baseline algorithm for comparison and simple deployments
- **FedProx**: Handles system heterogeneity common in industrial environments
- **FedDyn**: Addresses client drift in non-IID industrial data
- **FedNova**: Normalizes client contributions for fair aggregation

#### **2. Model Architecture Types (`ModelType`)**
```python
class ModelType(Enum):
    CNN = "cnn"          # Convolutional Neural Networks for spatial patterns
    LSTM = "lstm"        # Long Short-Term Memory for temporal sequences
    HYBRID = "hybrid"    # CNN-LSTM combination for complex patterns
```

**Usage:**
```python
from core import ModelType

# Type-safe model selection
model_type = ModelType.CNN
model = create_optimized_model(model_type, input_dim=10, num_classes=2)
```

**Industrial Context:**
- **CNN**: Sensor data pattern recognition, anomaly detection
- **LSTM**: Time-series prediction, equipment degradation modeling
- **Hybrid**: Complex industrial processes with spatial and temporal dependencies

#### **3. Data Distribution Strategies (`DataDistribution`)**
```python
class DataDistribution(Enum):
    IID = "iid"                    # Independent and identically distributed
    NON_IID_QUANTITY = "non_iid_quantity"  # Different data quantities
    NON_IID_FEATURE = "non_iid_feature"    # Different feature distributions
    NON_IID_LABEL = "non_iid_label"        # Different label distributions
```

**Usage:**
```python
from core import DataDistribution

# Type-safe distribution configuration
distribution = DataDistribution.NON_IID_LABEL
client_data = create_data_distribution(X_train, y_train, distribution_type=distribution)
```

**Industrial Reality:**
- **IID**: Ideal scenario for baseline comparison
- **Non-IID Quantity**: Different production volumes across factories
- **Non-IID Feature**: Different equipment types and sensor configurations
- **Non-IID Label**: Different failure patterns and maintenance schedules

#### **4. Experiment Status Tracking (`ExperimentStatus`)**
```python
class ExperimentStatus(Enum):
    PENDING = auto()      # Experiment configured, waiting to start
    RUNNING = auto()      # Currently executing federated rounds
    COMPLETED = auto()    # Successfully finished all rounds
    FAILED = auto()       # Error occurred during execution
    CANCELLED = auto()    # Manually terminated before completion
```

**Usage:**
```python
from core import ExperimentStatus

# Status tracking in orchestrator
if experiment.status == ExperimentStatus.RUNNING:
    continue_training()
elif experiment.status == ExperimentStatus.FAILED:
    handle_error_recovery()
```

#### **5. DataSite Status Management (`DataSiteStatus`)**
```python
class DataSiteStatus(Enum):
    OFFLINE = auto()      # Datasite not accessible
    ONLINE = auto()       # Available for federated learning
    BUSY = auto()         # Currently training or processing
    ERROR = auto()        # Error state requiring attention
```

**Usage:**
```python
from core import DataSiteStatus

# Industrial site monitoring
if datasite.status == DataSiteStatus.ONLINE:
    send_training_request(datasite)
elif datasite.status == DataSiteStatus.ERROR:
    alert_maintenance_team(datasite)
```

#### **6. Aggregation Methods (`AggregationMethod`)**
```python
class AggregationMethod(Enum):
    WEIGHTED_AVERAGE = "weighted_average"    # Weight by data quantity
    SIMPLE_AVERAGE = "simple_average"       # Equal weights for all clients
    MEDIAN = "median"                       # Byzantine-robust aggregation
```

**Usage:**
```python
from core import AggregationMethod

# Robust aggregation for industrial environments
method = AggregationMethod.MEDIAN  # Robust against faulty sensors
aggregated_params = aggregate_updates(client_updates, method=method)
```

#### **7. Privacy Protection Levels (`PrivacyLevel`)**
```python
class PrivacyLevel(Enum):
    NONE = "none"                           # No privacy protection
    BASIC = "basic"                         # Basic obfuscation
    DIFFERENTIAL_PRIVACY = "differential_privacy"  # DP guarantees
    SECURE_AGGREGATION = "secure_aggregation"      # Cryptographic protection
```

**Usage:**
```python
from core import PrivacyLevel

# Privacy configuration for industrial compliance
privacy_level = PrivacyLevel.DIFFERENTIAL_PRIVACY
secure_server = SecureFederatedServer(privacy_level=privacy_level)
```

### Why These Enumerations?

#### **Type Safety Benefits**
- **Compile-time Checking**: Catch invalid values during development
- **IDE Support**: Auto-completion and type hints
- **Refactoring Safety**: Easy to rename or modify enum values

#### **Industrial Compliance**
- **Audit Trail**: Clear documentation of configuration choices
- **Standardization**: Consistent terminology across teams and deployments
- **Validation**: Prevent invalid configuration combinations

---

## 3. Exception Hierarchy (`exceptions.py`)

### What is the Exception Hierarchy?
A structured exception system that provides specific error types for different failure modes in the federated learning system. This enables precise error handling and debugging for complex distributed industrial deployments.

### Exception Architecture

#### **Base Exception (`FederatedLearningError`)**
```python
class FederatedLearningError(Exception):
    """Base exception for federated learning system."""
    pass
```

**Purpose:**
- Root of all framework-specific exceptions
- Allows catching all framework errors with single handler
- Provides namespace separation from system exceptions

#### **Specialized Exception Types**

**Model-Related Errors (`ModelError`):**
```python
class ModelError(FederatedLearningError):
    """Exception raised for model-related errors."""
    pass
```

**Common Scenarios:**
- Model architecture incompatibility between sites
- Parameter dimension mismatches during aggregation
- Model loading/saving failures
- Forward pass computation errors

**DataSite Errors (`DataSiteError`):**
```python
class DataSiteError(FederatedLearningError):
    """Exception raised for datasite-related errors."""
    pass
```

**Industrial Scenarios:**
- Factory datasite connectivity failures
- Local data access permissions issues
- Industrial equipment sensor malfunctions
- Site-specific configuration errors

**Communication Errors (`CommunicationError`):**
```python
class CommunicationError(FederatedLearningError):
    """Exception raised for communication errors."""
    pass
```

**Distributed System Scenarios:**
- Network connectivity issues between sites
- PySyft authentication failures
- Message serialization/deserialization errors
- Timeout during distributed operations

**Aggregation Errors (`AggregationError`):**
```python
class AggregationError(FederatedLearningError):
    """Exception raised for aggregation errors."""
    pass
```

**Federated Learning Scenarios:**
- Insufficient client updates for aggregation
- Algorithm-specific aggregation failures
- Byzantine client detection errors
- Secure aggregation cryptographic failures

**Experiment Errors (`ExperimentError`):**
```python
class ExperimentError(FederatedLearningError):
    """Exception raised for experiment execution errors."""
    pass
```

**Research/Production Scenarios:**
- Experiment configuration validation failures
- Resource allocation errors
- Early stopping criteria violations
- Results collection and storage errors

**Configuration Errors (`ConfigurationError`):**
```python
class ConfigurationError(FederatedLearningError):
    """Exception raised for configuration errors."""
    pass
```

**System Setup Scenarios:**
- Invalid YAML configuration files
- Missing required configuration parameters
- Incompatible configuration combinations
- Environment variable resolution failures

**Security Errors (`SecurityError`):**
```python
class SecurityError(FederatedLearningError):
    """Exception raised for security-related errors."""
    pass
```

**Industrial Security Scenarios:**
- Differential privacy budget exhaustion
- Byzantine client attack detection
- Authentication and authorization failures
- Secure communication channel establishment errors

**Data Errors (`DataError`):**
```python
class DataError(FederatedLearningError):
    """Exception raised for data-related errors."""
    pass
```

**Data Pipeline Scenarios:**
- Industrial sensor data validation failures
- Data format incompatibility across sites
- Missing or corrupted training data
- Data distribution imbalance issues

### Exception Usage Patterns

#### **Specific Error Handling**
```python
from core import ModelError, DataSiteError, CommunicationError

try:
    # Federated learning operations
    model_params = aggregate_client_updates(client_updates)
    
except ModelError as e:
    logger.error(f"Model aggregation failed: {e}")
    # Handle model-specific recovery
    
except DataSiteError as e:
    logger.error(f"DataSite operation failed: {e}")
    # Handle site-specific recovery
    
except CommunicationError as e:
    logger.error(f"Communication failed: {e}")
    # Handle network recovery
```

#### **General Framework Error Handling**
```python
from core import FederatedLearningError

try:
    # Complex federated learning workflow
    result = run_federated_experiment(config)
    
except FederatedLearningError as e:
    logger.error(f"Federated learning error: {e}")
    # General framework error handling
    
except Exception as e:
    logger.error(f"Unexpected system error: {e}")
    # Handle non-framework errors
```

#### **Industrial Error Recovery**
```python
from core import DataSiteError, CommunicationError

def robust_industrial_training():
    failed_sites = []
    successful_sites = []
    
    for site in industrial_sites:
        try:
            result = train_on_site(site)
            successful_sites.append((site, result))
            
        except DataSiteError:
            # Log equipment failure, schedule maintenance
            failed_sites.append(site)
            schedule_maintenance(site)
            
        except CommunicationError:
            # Log network issue, retry with backup connection
            failed_sites.append(site)
            attempt_backup_connection(site)
    
    if len(successful_sites) >= minimum_participants:
        return aggregate_results(successful_sites)
    else:
        raise ExperimentError("Insufficient participants for federated learning")
```

### Why This Exception Hierarchy?

#### **Debugging Benefits**
- **Precise Error Identification**: Know exactly which component failed
- **Contextual Error Handling**: Different recovery strategies for different error types
- **Stack Trace Clarity**: Clear error propagation through system layers

#### **Industrial Robustness**
- **Fault Tolerance**: Continue operation despite individual component failures
- **Maintenance Scheduling**: Automatic error reporting for industrial equipment
- **Compliance Logging**: Detailed error tracking for industrial audits

---

## 4. Core Module Integration (`__init__.py`)

### What is Core Module Integration?
The `__init__.py` file provides a unified import interface for all core components, making it easy to access interfaces, enumerations, and exceptions throughout the federated learning framework.

### Unified Import Interface

#### **Complete Import Structure**
```python
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
```

#### **Simplified Usage Patterns**
```python
# Single import for all core components
from core import (
    ModelInterface, FederatedAlgorithm, ModelType, 
    ModelError, ConfigurationError
)

# Use type-safe enumerations
algorithm = FederatedAlgorithm.FEDAVG
model_type = ModelType.CNN

# Implement interfaces
class IndustrialCNNModel(ModelInterface):
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return self.state_dict()
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(parameters)

# Handle specific exceptions
try:
    model = IndustrialCNNModel(config)
except ConfigurationError as e:
    logger.error(f"Model configuration failed: {e}")
```

### Public API Definition

#### **Explicit Exports (`__all__`)**
```python
__all__ = [
    # Interfaces - 6 core contracts
    'ModelInterface', 'FederatedAlgorithmInterface', 'DataSiteInterface', 
    'OrchestratorInterface', 'MetricsCollectorInterface', 'CommunicationInterface',
    
    # Enums - 7 type-safe constants
    'FederatedAlgorithm', 'ModelType', 'DataDistribution',
    'ExperimentStatus', 'DataSiteStatus', 'AggregationMethod', 'PrivacyLevel',
    
    # Exceptions - 9 error types
    'FederatedLearningError', 'ModelError', 'DataSiteError',
    'CommunicationError', 'AggregationError', 'ExperimentError',
    'ConfigurationError', 'SecurityError', 'DataError'
]
```

**Benefits:**
- **Clear Public API**: Only intended exports are accessible
- **IDE Support**: Auto-completion works correctly
- **Documentation**: Clear indication of supported functionality

---

## 5. Integration with Framework Components

### Component Implementation Patterns

#### **Model Implementation**
```python
from core import ModelInterface, ModelType, ModelError

class OptimizedCNNModel(nn.Module, ModelInterface):
    def __init__(self, input_dim: int = 10, num_classes: int = 2):
        super().__init__()
        self.model_type = ModelType.CNN
        # CNN architecture implementation
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Implementation of ModelInterface contract."""
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Implementation of ModelInterface contract."""
        try:
            self.load_state_dict(parameters)
        except RuntimeError as e:
            raise ModelError(f"Parameter loading failed: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of ModelInterface contract."""
        # Forward pass implementation
        return self.classifier(features)
```

#### **Algorithm Implementation**
```python
from core import FederatedAlgorithmInterface, FederatedAlgorithm, AggregationError

class FedAvgAlgorithm(FederatedAlgorithmInterface):
    def __init__(self):
        self.algorithm_type = FederatedAlgorithm.FEDAVG
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Implement federated averaging aggregation."""
        try:
            # Extract model parameters and weights
            parameters_list = [update['parameters'] for update in client_updates]
            weights = [update['num_samples'] for update in client_updates]
            
            # Perform weighted averaging
            aggregated = {}
            total_weight = sum(weights)
            
            for param_name in parameters_list[0].keys():
                weighted_sum = sum(
                    params[param_name] * weight 
                    for params, weight in zip(parameters_list, weights)
                )
                aggregated[param_name] = weighted_sum / total_weight
                
            return aggregated
            
        except Exception as e:
            raise AggregationError(f"FedAvg aggregation failed: {e}")
    
    def configure_client_training(self, round_num: int) -> Dict[str, Any]:
        """Configure FedAvg client training (no special configuration)."""
        return {'algorithm': 'fedavg', 'round': round_num}
```

#### **Communication Implementation**
```python
from core import CommunicationInterface, CommunicationError, DataSiteStatus

class PySyftCommunicationManager(CommunicationInterface):
    def __init__(self):
        self.connected_sites = {}
        self.site_status = {}
    
    def connect_to_datasite(self, datasite_url: str, credentials: Dict[str, str]) -> bool:
        """Implement PySyft datasite connection."""
        try:
            client = sy.login(
                url=datasite_url, 
                email=credentials['email'],
                password=credentials['password']
            )
            
            site_id = self._extract_site_id(datasite_url)
            self.connected_sites[site_id] = client
            self.site_status[site_id] = DataSiteStatus.ONLINE
            return True
            
        except Exception as e:
            raise CommunicationError(f"Failed to connect to {datasite_url}: {e}")
    
    def send_model(self, model_parameters: Dict[str, torch.Tensor], 
                   datasite_id: str) -> bool:
        """Send model parameters to datasite."""
        if datasite_id not in self.connected_sites:
            raise CommunicationError(f"Not connected to datasite: {datasite_id}")
        
        try:
            client = self.connected_sites[datasite_id]
            # Implement PySyft model sending
            success = client.send_model_parameters(model_parameters)
            return success
            
        except Exception as e:
            self.site_status[datasite_id] = DataSiteStatus.ERROR
            raise CommunicationError(f"Failed to send model to {datasite_id}: {e}")
```

### Exception Handling Integration

#### **Orchestrator Error Management**
```python
from core import (
    ExperimentStatus, ExperimentError, DataSiteError, 
    CommunicationError, ModelError
)

class FederatedExperimentOrchestrator:
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run federated experiment with comprehensive error handling."""
        try:
            # Set experiment status
            self.experiments[experiment_id]['status'] = ExperimentStatus.RUNNING
            
            # Execute federated rounds
            for round_num in range(max_rounds):
                try:
                    round_results = self._run_federated_round(round_num)
                    
                except DataSiteError as e:
                    # Handle individual site failures
                    logger.warning(f"DataSite failure in round {round_num}: {e}")
                    self._handle_site_failure(e, round_num)
                    
                except CommunicationError as e:
                    # Handle network issues
                    logger.warning(f"Communication error in round {round_num}: {e}")
                    self._retry_with_backup_sites(round_num)
                    
                except ModelError as e:
                    # Handle model aggregation failures
                    logger.error(f"Model error in round {round_num}: {e}")
                    raise ExperimentError(f"Experiment failed due to model error: {e}")
            
            # Mark experiment as completed
            self.experiments[experiment_id]['status'] = ExperimentStatus.COMPLETED
            return self._collect_experiment_results(experiment_id)
            
        except ExperimentError:
            # Re-raise experiment errors
            self.experiments[experiment_id]['status'] = ExperimentStatus.FAILED
            raise
            
        except Exception as e:
            # Handle unexpected errors
            self.experiments[experiment_id]['status'] = ExperimentStatus.FAILED
            raise ExperimentError(f"Unexpected experiment failure: {e}")
```

---

## 6. Production Deployment Guidelines

### Design Pattern Implementation

#### **Dependency Injection with Interfaces**
```python
from core import ModelInterface, FederatedAlgorithmInterface, CommunicationInterface

class FederatedServer:
    def __init__(self, 
                 model: ModelInterface,
                 algorithm: FederatedAlgorithmInterface,
                 communication: CommunicationInterface):
        self.model = model
        self.algorithm = algorithm
        self.communication = communication
    
    # Server implementation uses interfaces, not concrete classes
```

**Benefits:**
- **Testability**: Easy to inject mock implementations for testing
- **Flexibility**: Swap implementations without changing server code
- **Maintainability**: Clear dependencies and reduced coupling

#### **Type-Safe Configuration**
```python
from core import FederatedAlgorithm, ModelType, PrivacyLevel

@dataclass
class ExperimentConfig:
    algorithm: FederatedAlgorithm
    model_type: ModelType
    privacy_level: PrivacyLevel
    max_rounds: int
    
    def __post_init__(self):
        # Validation using enums
        if not isinstance(self.algorithm, FederatedAlgorithm):
            raise ConfigurationError("Invalid algorithm type")
```

### Error Recovery Strategies

#### **Industrial Resilience Patterns**
```python
from core import DataSiteError, CommunicationError, DataSiteStatus

class IndustrialResilientOrchestrator:
    def __init__(self):
        self.retry_policies = {
            DataSiteError: self._handle_equipment_failure,
            CommunicationError: self._handle_network_failure
        }
    
    def _handle_equipment_failure(self, site_id: str, error: DataSiteError):
        """Handle industrial equipment failures."""
        # Log maintenance request
        self.maintenance_scheduler.schedule_inspection(site_id)
        
        # Mark site as requiring attention
        self.site_status[site_id] = DataSiteStatus.ERROR
        
        # Continue with remaining healthy sites
        return self._get_healthy_sites()
    
    def _handle_network_failure(self, site_id: str, error: CommunicationError):
        """Handle network connectivity issues."""
        # Attempt backup connection methods
        backup_success = self._try_backup_connection(site_id)
        
        if backup_success:
            self.site_status[site_id] = DataSiteStatus.ONLINE
        else:
            # Temporary removal from training
            self.site_status[site_id] = DataSiteStatus.OFFLINE
            self._schedule_reconnection_attempt(site_id)
```

### Monitoring and Observability

#### **Status Tracking Integration**
```python
from core import ExperimentStatus, DataSiteStatus

class SystemMonitor:
    def collect_system_status(self) -> Dict[str, Any]:
        """Collect comprehensive system status."""
        return {
            'experiments': {
                exp_id: exp['status'] for exp_id, exp in self.experiments.items()
            },
            'datasites': {
                site_id: status for site_id, status in self.site_status.items()
            },
            'healthy_sites': len([
                s for s in self.site_status.values() 
                if s == DataSiteStatus.ONLINE
            ]),
            'failed_experiments': len([
                e for e in self.experiments.values() 
                if e['status'] == ExperimentStatus.FAILED
            ])
        }
```

---

## 7. Development Best Practices

### Interface Implementation Guidelines

#### **Contract Compliance**
```python
# ✅ Good: Proper interface implementation
class GoodModelImplementation(ModelInterface):
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Proper type hints and documentation."""
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Proper error handling."""
        try:
            self.load_state_dict(parameters)
        except RuntimeError as e:
            raise ModelError(f"Parameter loading failed: {e}")

# ❌ Bad: Missing proper error handling
class BadModelImplementation(ModelInterface):
    def get_parameters(self):  # Missing type hints
        return self.state_dict()  # Might fail without proper error handling
    
    def set_parameters(self, parameters):  # Missing type hints
        self.load_state_dict(parameters)  # Raw exceptions leak
```

#### **Enum Usage Patterns**
```python
from core import FederatedAlgorithm, ModelType

# ✅ Good: Type-safe enum usage
def create_experiment(algorithm: FederatedAlgorithm, model: ModelType):
    if algorithm == FederatedAlgorithm.FEDPROX:
        # Algorithm-specific configuration
        pass

# ❌ Bad: String-based configuration (error-prone)
def create_experiment_bad(algorithm: str, model: str):
    if algorithm == "fedprox":  # Typo-prone, no IDE support
        pass
```

#### **Exception Handling Best Practices**
```python
from core import FederatedLearningError, ModelError, DataSiteError

# ✅ Good: Specific exception handling
try:
    result = complex_federated_operation()
except ModelError as e:
    logger.error(f"Model-specific error: {e}")
    return handle_model_recovery()
except DataSiteError as e:
    logger.error(f"DataSite error: {e}")
    return handle_site_recovery()
except FederatedLearningError as e:
    logger.error(f"General framework error: {e}")
    return handle_general_recovery()

# ❌ Bad: Generic exception handling loses context
try:
    result = complex_federated_operation()
except Exception as e:  # Too broad, loses specific error context
    logger.error(f"Something failed: {e}")
    return generic_recovery()  # Can't provide appropriate recovery
```

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Framework Version**: 2.0  
**Architecture Pattern**: Clean Architecture with Abstract Interfaces  
**Industrial Application**: Predictive Maintenance in Manufacturing IoT Systems
