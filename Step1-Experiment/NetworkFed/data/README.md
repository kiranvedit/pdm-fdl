# Federated Data Integration Layer

This directory contains the data integration layer that connects the federated learning framework with the existing processed data infrastructure. It provides sophisticated data distribution strategies, federated dataset management, and datasite data handling for industrial IoT predictive maintenance scenarios.

## Architecture Overview

The data integration layer implements a comprehensive pipeline for federated data management:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Integration Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │  Processed Data │    │  Distribution   │    │  DataSite    │  │
│  │   (shared/)     │───▶│   Strategies    │───▶│  Management  │  │
│  │   - Tabular     │    │   - IID         │    │   - Local    │  │
│  │   - Sequences   │    │   - Non-IID     │    │   - Training │  │
│  │   - Multiclass  │    │   - Dirichlet   │    │   - Testing  │  │
│  └─────────────────┘    └─────────────────┘    └──────────────┘  │
│                                 │                               │
│  ┌─────────────────────────────┬┘                               │
│  │                             ▼                               │
│  │    ┌─────────────────────────────────────────────────────┐   │
│  │    │         Federated Dataset Management                │   │
│  │    │  - PyTorch Dataset Integration                      │   │
│  │    │  - DataLoader Generation                            │   │
│  │    │  - Statistics Collection                            │   │
│  │    │  - Metadata Management                              │   │
│  │    └─────────────────────────────────────────────────────┘   │
│  │                                                             │
│  └─────────────────────────────────────────────────────────────┘
```

---

## 1. Core Components Overview

### **Module Exports (`__init__.py`)**
```python
from .data_integration import (
    FederatedDataDistributor,
    FederatedDataset, 
    DataSiteDataManager
)
```

The data module provides three primary components:
- **FederatedDataDistributor**: Manages data distribution across multiple datasites
- **FederatedDataset**: PyTorch-compatible dataset for federated learning
- **DataSiteDataManager**: Handles datasite-specific data operations

---

## 2. Federated Data Distributor (`FederatedDataDistributor`)

### **What is the Federated Data Distributor?**
The `FederatedDataDistributor` is the central component responsible for taking processed industrial IoT data and distributing it across multiple federated learning datasites according to various strategies that simulate real-world industrial scenarios.

### **Core Functionality**

#### **Initialization and Metadata Integration**
```python
distributor = FederatedDataDistributor(processed_data_path="/path/to/shared/processed_data")
```

**Features:**
- Integrates with existing `shared/utils/step1_data_utils.py` data pipeline
- Automatically loads metadata from processed data directory
- Supports three data types: `tabular`, `sequences`, `multiclass`
- Maintains compatibility with existing project data structure

#### **Data Distribution Strategies**

**1. IID Distribution (`'iid'`)**
```python
datasets = distributor.create_datasite_datasets(
    num_datasites=5,
    distribution_strategy='iid',
    data_type='tabular'
)
```

**Purpose:**
- **Baseline Comparison**: Ideal scenario for federated learning
- **Algorithm Validation**: Standard benchmark for new algorithms
- **Performance Upper Bound**: Best-case scenario for federated learning

**Implementation:**
- Random shuffle of all data samples
- Equal distribution across all datasites
- Maintains original label proportions at each site

**Industrial Context:**
- Represents uniform manufacturing processes
- Identical equipment across all factory sites
- Standardized operational procedures

**2. Non-IID Label Distribution (`'non_iid_label'`)**
```python
datasets = distributor.create_datasite_datasets(
    num_datasites=8,
    distribution_strategy='non_iid_label',
    data_type='multiclass',
    alpha=0.1  # Lower alpha = more skewed distribution
)
```

**Purpose:**
- **Industrial Reality**: Different failure patterns across sites
- **Specialization**: Sites with specific equipment types
- **Challenge Testing**: Algorithm robustness under data heterogeneity

**Implementation:**
- Uses Dirichlet distribution with configurable alpha parameter
- Each class distributed according to Dirichlet probabilities
- Lower alpha values create more skewed distributions

**Alpha Parameter Effects:**
- `alpha = 0.1`: Highly skewed (each site specializes in few failure types)
- `alpha = 0.5`: Moderately skewed (some specialization)
- `alpha = 1.0`: Balanced distribution (closer to IID)

**Industrial Scenarios:**
- **Manufacturing Lines**: Different products, different failure modes
- **Equipment Types**: Pumps vs. motors vs. compressors
- **Operational Conditions**: High-stress vs. normal vs. light-duty environments

**3. Non-IID Quantity Distribution (`'non_iid_quantity'`)**
```python
datasets = distributor.create_datasite_datasets(
    num_datasites=6,
    distribution_strategy='non_iid_quantity',
    data_type='sequences',
    alpha=0.3  # Controls quantity imbalance
)
```

**Purpose:**
- **Resource Variation**: Different data collection capabilities
- **Scale Differences**: Large vs. small manufacturing facilities
- **Temporal Variation**: Different operational histories

**Implementation:**
- Dirichlet distribution determines data quantity per site
- Label distributions remain balanced within each site
- Some sites get significantly more/less data

**Industrial Reality:**
- **Facility Size**: Large plants vs. small workshops
- **Sensor Density**: Comprehensive vs. limited monitoring
- **Operational History**: Established vs. new facilities

#### **Data Type Support**

**Tabular Data Processing**
```python
# Load tabular features for traditional ML models
X_train, y_train, X_val, y_val, X_test, y_test = distributor._load_tabular_data(data_path)
```

**Features:**
- Standard tabular format (samples × features)
- Supports validation splits
- Compatible with CNN models for pattern recognition
- Fallback loading from project data loader

**Use Cases:**
- Sensor reading aggregations
- Equipment parameter summaries
- Maintenance record features

**Sequence Data Processing**
```python
# Load temporal sequences for LSTM/time-series models
X_train, y_train, X_val, y_val, X_test, y_test = distributor._load_sequence_data(data_path)
```

**Features:**
- 3D tensor format (samples × timesteps × features)
- Temporal sequence preservation during distribution
- Supports variable-length sequences
- LSTM/RNN model compatibility

**Use Cases:**
- Vibration signal sequences
- Temperature progression patterns
- Performance degradation trajectories

**Multiclass Data Processing**
```python
# Load multiclass classification data for detailed failure analysis
X_train, y_train, X_val, y_val, X_test, y_test = distributor._load_multiclass_data(data_path)
```

**Features:**
- Multiple failure type classification
- Complex label relationships
- Specialized multiclass label files
- Advanced failure pattern recognition

**Use Cases:**
- Detailed failure type prediction
- Root cause analysis
- Multi-component failure detection

### **Advanced Features**

#### **Intelligent Data Splitting**
```python
# Automatic train/validation split preservation
train_ratio = len(X_train) / total_train_val_samples
```

**Benefits:**
- Maintains original train/validation proportions
- Ensures consistent evaluation across datasites
- Preserves temporal order in sequence data

#### **Comprehensive Statistics Collection**
```python
stats = distributor.get_data_statistics(datasite_datasets)
```

**Statistical Metrics:**
- Total samples across all datasites
- Samples per individual datasite
- Label distribution analysis per datasite
- Feature statistics and distributions

**Industrial Insights:**
- Data imbalance detection
- Site specialization analysis
- Quality assurance validation

#### **Complete Test Data Access**
```python
test_data = distributor.get_complete_test_data(data_type='tabular')
```

**Features:**
- Centralized test set for fair evaluation
- Consistent across all datasites
- Supports all data types
- Maintains evaluation integrity

---

## 3. Federated Dataset (`FederatedDataset`)

### **What is the Federated Dataset?**
`FederatedDataset` is a PyTorch-compatible dataset class that wraps distributed data for individual datasites, providing seamless integration with PyTorch's training pipeline while maintaining federated learning metadata.

### **Core Functionality**

#### **PyTorch Integration**
```python
# Create dataset from distributed data
dataset = FederatedDataset(
    X=site_X,  # Training features for this datasite
    y=site_y,  # Training labels for this datasite  
    metadata=metadata,  # Comprehensive datasite information
    datasite_id="factory_01"
)

# Standard PyTorch usage
dataloader = dataset.get_dataloader(batch_size=32, shuffle=True)
for batch_X, batch_y in dataloader:
    # Standard PyTorch training loop
    model_output = model(batch_X)
    loss = criterion(model_output, batch_y)
```

**PyTorch Compatibility:**
- Implements standard `Dataset` interface
- Automatic tensor conversion (FloatTensor for features, LongTensor for labels)
- Built-in DataLoader generation
- Batch processing support

#### **Metadata Management**
```python
# Access comprehensive datasite information
metadata = {
    'datasite_id': 'factory_01',
    'distribution_strategy': 'non_iid_label',
    'alpha': 0.1,
    'data_type': 'multiclass',
    'train_samples': 1250,
    'val_samples': 312,
    'test_samples': 500,
    'features': [0, 1, 2, ..., 9],  # Feature indices
    'classes': [0, 1, 2, 3, 4],     # Available classes at this site
    'X_val': validation_features,    # Validation data
    'y_val': validation_labels,     # Validation labels
    'X_test': test_features,        # Complete test data
    'y_test': test_labels          # Complete test labels
}
```

**Metadata Benefits:**
- Complete traceability of data distribution
- Validation and test data accessibility
- Algorithm parameter tracking
- Site-specific information storage

#### **Dataset Statistics and Analysis**
```python
# Get detailed dataset statistics
stats = dataset.get_statistics()
```

**Statistical Information:**
```python
{
    'num_samples': 1250,
    'num_features': 10,
    'num_classes': 5,
    'feature_means': [0.45, 0.32, ...],  # Mean per feature
    'feature_stds': [0.12, 0.08, ...],   # Std deviation per feature
    'class_distribution': {
        '0': 450,  # Normal operation
        '1': 320,  # Heat dissipation failure
        '2': 280,  # Overstrain failure  
        '3': 150,  # Power failure
        '4': 50    # Tool wear failure
    }
}
```

**Industrial Applications:**
- **Data Quality Assessment**: Identify anomalous distributions
- **Site Characterization**: Understand equipment specialization
- **Training Optimization**: Adjust learning parameters per site

#### **Feature and Target Information**
```python
# Access feature and target metadata
feature_names = dataset.get_feature_names()  # ['Temperature', 'Vibration', ...]
target_names = dataset.get_target_names()    # ['Normal', 'Failure_Type_1', ...]
```

---

## 4. DataSite Data Manager (`DataSiteDataManager`)

### **What is the DataSite Data Manager?**
`DataSiteDataManager` provides a high-level interface for managing all data operations at a specific industrial datasite, including training data access, test data handling, and site-specific analytics.

### **Industrial Datasite Operations**

#### **Training Data Management**
```python
# Initialize datasite manager for a specific factory
manager = DataSiteDataManager(
    datasite_id="factory_automotive_01",
    train_dataset=federated_dataset
)

# Get training data for local model training
train_loader = manager.get_training_data(batch_size=64)

# Standard federated learning training loop
for epoch in range(local_epochs):
    for batch_X, batch_y in train_loader:
        # Local model training
        optimizer.zero_grad()
        outputs = local_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

#### **Evaluation and Testing**
```python
# Get test data for model evaluation
test_loader = manager.get_test_data(batch_size=64)

# Evaluate local model performance
local_model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = local_model(batch_X)
        predictions = torch.argmax(outputs, dim=1)
        # Calculate metrics
```

#### **Site Information and Analytics**
```python
# Get comprehensive datasite information
site_info = manager.get_data_info()
```

**Site Information Structure:**
```python
{
    'datasite_id': 'factory_automotive_01',
    'train_statistics': {
        'num_samples': 1250,
        'num_features': 10,
        'num_classes': 5,
        'feature_means': [...],
        'feature_stds': [...],
        'class_distribution': {...}
    },
    'test_statistics': {
        # Similar structure for test data
    },
    'feature_names': ['Air_temperature', 'Process_temperature', ...],
    'target_names': ['No_Failure', 'Heat_Dissipation_Failure', ...]
}
```

#### **Data Sampling for Research**
```python
# Sample subset of data for analysis or lightweight training
sample_X, sample_y = manager.sample_data(num_samples=100)
```

**Use Cases:**
- **Quick Analysis**: Fast statistical analysis without full dataset
- **Model Debugging**: Test with small data samples
- **Communication Efficiency**: Send data samples for analysis

### **Industrial Integration Patterns**

#### **Factory Equipment Integration**
```python
class IndustrialDataSite:
    def __init__(self, site_id: str, equipment_config: Dict):
        self.data_manager = DataSiteDataManager(site_id, federated_dataset)
        self.equipment_config = equipment_config
        
    def collect_sensor_data(self) -> torch.Tensor:
        """Simulate real-time sensor data collection."""
        # Interface with industrial sensors
        sensor_data = self.read_equipment_sensors()
        return torch.FloatTensor(sensor_data)
    
    def train_local_model(self, global_model: torch.nn.Module) -> Dict:
        """Train model on local industrial data."""
        train_loader = self.data_manager.get_training_data()
        
        # Local training with equipment-specific data
        for epoch in range(self.local_epochs):
            for batch_X, batch_y in train_loader:
                # Training implementation
                pass
        
        return {
            'model_parameters': global_model.state_dict(),
            'training_samples': len(train_loader.dataset),
            'training_loss': final_loss,
            'site_id': self.data_manager.datasite_id
        }
```

#### **Multi-Site Coordination**
```python
class FederatedIndustrialNetwork:
    def __init__(self, datasite_datasets: Dict[str, FederatedDataset]):
        self.site_managers = {
            site_id: DataSiteDataManager(site_id, dataset)
            for site_id, dataset in datasite_datasets.items()
        }
    
    def collect_site_statistics(self) -> Dict[str, Any]:
        """Collect statistics from all industrial sites."""
        return {
            site_id: manager.get_data_info()
            for site_id, manager in self.site_managers.items()
        }
    
    def coordinate_federated_training(self, global_model: torch.nn.Module):
        """Coordinate training across all industrial sites."""
        site_updates = {}
        
        for site_id, manager in self.site_managers.items():
            # Send global model to site
            local_model = copy.deepcopy(global_model)
            
            # Train locally at each site
            train_loader = manager.get_training_data()
            site_updates[site_id] = self.train_at_site(local_model, train_loader)
        
        return site_updates
```

---

## 5. Data Integration Workflow

### **Complete Usage Example**

#### **Step 1: Initialize Data Distribution**
```python
from data import FederatedDataDistributor

# Initialize with processed data path
distributor = FederatedDataDistributor(
    processed_data_path="/path/to/shared/processed_data"
)

# Create federated datasets for industrial sites
datasite_datasets = distributor.create_datasite_datasets(
    num_datasites=8,                    # 8 manufacturing sites
    distribution_strategy='non_iid_label',  # Different failure patterns per site
    data_type='multiclass',             # Detailed failure classification
    alpha=0.2                          # Moderate specialization
)
```

#### **Step 2: Analyze Data Distribution**
```python
# Get distribution statistics
stats = distributor.get_data_statistics(datasite_datasets)

print(f"Total samples: {stats['total_samples']}")
print(f"Number of sites: {stats['num_datasites']}")

# Analyze per-site distributions
for site_id, samples in stats['samples_per_datasite'].items():
    print(f"{site_id}: {samples} samples")
    
    # Label distribution analysis
    labels = stats['label_distribution_per_datasite'][site_id]
    print(f"  Label distribution: {labels}")
```

#### **Step 3: Create Site Managers**
```python
from data import DataSiteDataManager

# Create managers for each industrial site
site_managers = {}
for site_id, dataset in datasite_datasets.items():
    site_managers[site_id] = DataSiteDataManager(
        datasite_id=site_id,
        train_dataset=dataset
    )
```

#### **Step 4: Training Integration**
```python
# Federated learning training loop
for round_num in range(max_rounds):
    site_updates = []
    
    # Train at each site
    for site_id, manager in site_managers.items():
        # Get local training data
        train_loader = manager.get_training_data(batch_size=32)
        
        # Local training
        local_model = copy.deepcopy(global_model)
        local_optimizer = torch.optim.Adam(local_model.parameters())
        
        for epoch in range(local_epochs):
            for batch_X, batch_y in train_loader:
                local_optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                local_optimizer.step()
        
        # Collect update
        site_updates.append({
            'site_id': site_id,
            'parameters': local_model.state_dict(),
            'num_samples': len(train_loader.dataset)
        })
    
    # Aggregate updates (implement aggregation logic)
    global_model = aggregate_model_updates(site_updates)
```

#### **Step 5: Evaluation**
```python
# Get complete test data for evaluation
test_data = distributor.get_complete_test_data(data_type='multiclass')
X_test, y_test = test_data['X_test'], test_data['y_test']

# Create test dataset
test_dataset = FederatedDataset(X_test, y_test, {})
test_loader = test_dataset.get_dataloader(batch_size=64, shuffle=False)

# Evaluate global model
global_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = global_model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f'Global Model Accuracy: {accuracy:.2f}%')
```

---

## 6. Industrial Applications and Use Cases

### **Manufacturing Scenario Examples**

#### **Automotive Manufacturing Network**
```python
# Simulate automotive manufacturing sites with different specializations
automotive_datasets = distributor.create_datasite_datasets(
    num_datasites=6,
    distribution_strategy='non_iid_label',
    data_type='multiclass',
    alpha=0.1  # High specialization
)

# Site specializations:
# Site 1: Engine manufacturing (power failure expertise)
# Site 2: Transmission (overstrain failure expertise)  
# Site 3: Assembly line (heat dissipation expertise)
# Site 4: Quality control (tool wear expertise)
# Site 5: Testing facility (general failures)
# Site 6: Maintenance depot (all failure types)
```

#### **Chemical Processing Plants**
```python
# Chemical plants with different equipment scales
chemical_datasets = distributor.create_datasite_datasets(
    num_datasites=4,
    distribution_strategy='non_iid_quantity', 
    data_type='sequences',
    alpha=0.3  # Varying plant sizes
)

# Plant characteristics:
# Plant 1: Large refinery (60% of data)
# Plant 2: Medium petrochemical (25% of data)
# Plant 3: Small specialty chemicals (10% of data)
# Plant 4: Research facility (5% of data)
```

#### **Electronics Manufacturing**
```python
# Electronics sites with temporal data patterns
electronics_datasets = distributor.create_datasite_datasets(
    num_datasites=10,
    distribution_strategy='iid',
    data_type='tabular',
    alpha=1.0  # Standardized processes
)

# All sites follow similar processes but contribute to robustness
```

### **Real-World Integration Considerations**

#### **Data Privacy and Security**
```python
class PrivacyAwareDataManager(DataSiteDataManager):
    def __init__(self, datasite_id: str, train_dataset: FederatedDataset, 
                 privacy_budget: float = 1.0):
        super().__init__(datasite_id, train_dataset)
        self.privacy_budget = privacy_budget
    
    def get_training_data(self, batch_size: int = 32, 
                         add_noise: bool = True) -> TorchDataLoader:
        """Get training data with optional differential privacy."""
        if add_noise and self.privacy_budget > 0:
            # Add differential privacy noise
            return self._add_dp_noise(
                super().get_training_data(batch_size), 
                self.privacy_budget
            )
        return super().get_training_data(batch_size)
```

#### **Equipment Integration**
```python
class EquipmentDataInterface:
    def __init__(self, data_manager: DataSiteDataManager, 
                 sensor_config: Dict[str, Any]):
        self.data_manager = data_manager
        self.sensor_config = sensor_config
    
    def collect_real_time_data(self) -> torch.Tensor:
        """Interface with actual industrial equipment."""
        # Collect from PLCs, SCADA systems, etc.
        sensor_readings = self.read_industrial_sensors()
        
        # Apply same preprocessing as training data
        processed_data = self.preprocess_sensor_data(sensor_readings)
        
        return torch.FloatTensor(processed_data)
    
    def predict_maintenance_needs(self, model: torch.nn.Module) -> Dict:
        """Real-time prediction for maintenance scheduling."""
        real_time_data = self.collect_real_time_data()
        
        with torch.no_grad():
            prediction = model(real_time_data.unsqueeze(0))
            confidence = torch.softmax(prediction, dim=1)
        
        return {
            'prediction': torch.argmax(prediction).item(),
            'confidence': torch.max(confidence).item(),
            'timestamp': datetime.now(),
            'equipment_id': self.sensor_config['equipment_id']
        }
```

---

## 7. Performance Optimization and Best Practices

### **Memory Efficiency**

#### **Lazy Loading Patterns**
```python
class EfficientFederatedDataset(FederatedDataset):
    def __init__(self, data_path: str, metadata: Dict[str, Any]):
        # Store only path, load data on demand
        self.data_path = data_path
        self.metadata = metadata
        self._X = None
        self._y = None
    
    @property
    def X(self):
        if self._X is None:
            self._X = torch.FloatTensor(np.load(self.data_path + '_X.npy'))
        return self._X
    
    @property  
    def y(self):
        if self._y is None:
            self._y = torch.LongTensor(np.load(self.data_path + '_y.npy'))
        return self._y
```

#### **Batch Processing Optimization**
```python
def optimize_batch_size(dataset: FederatedDataset, 
                       model: torch.nn.Module,
                       max_memory_gb: float = 4.0) -> int:
    """Automatically determine optimal batch size based on available memory."""
    # Start with small batch and increase until memory limit
    for batch_size in [16, 32, 64, 128, 256, 512]:
        try:
            # Test memory usage with this batch size
            sample_batch = dataset.X[:batch_size]
            _ = model(sample_batch)
            
            # Check GPU memory usage if available
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                if memory_used > max_memory_gb:
                    return batch_size // 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return batch_size // 2
    
    return 256  # Default fallback
```

### **Data Quality Assurance**

#### **Distribution Validation**
```python
def validate_data_distribution(datasite_datasets: Dict[str, FederatedDataset],
                              min_samples_per_site: int = 50,
                              min_classes_per_site: int = 2) -> Dict[str, bool]:
    """Validate that data distribution meets quality requirements."""
    validation_results = {}
    
    for site_id, dataset in datasite_datasets.items():
        # Check minimum samples
        has_min_samples = len(dataset) >= min_samples_per_site
        
        # Check minimum classes
        unique_classes = len(torch.unique(dataset.y))
        has_min_classes = unique_classes >= min_classes_per_site
        
        # Check for data balance
        class_counts = torch.bincount(dataset.y)
        max_imbalance = torch.max(class_counts) / torch.min(class_counts)
        is_reasonably_balanced = max_imbalance <= 10.0  # 10:1 ratio limit
        
        validation_results[site_id] = {
            'has_min_samples': has_min_samples,
            'has_min_classes': has_min_classes,
            'is_reasonably_balanced': is_reasonably_balanced,
            'passes_validation': all([has_min_samples, has_min_classes, is_reasonably_balanced])
        }
    
    return validation_results
```

#### **Industrial Data Quality Checks**
```python
def industrial_data_quality_check(dataset: FederatedDataset) -> Dict[str, Any]:
    """Perform industrial-specific data quality validation."""
    X, y = dataset.X, dataset.y
    
    quality_report = {
        'missing_values': torch.isnan(X).sum().item(),
        'infinite_values': torch.isinf(X).sum().item(),
        'negative_values': (X < 0).sum().item(),
        'zero_variance_features': (torch.var(X, dim=0) == 0).sum().item(),
        'outliers_detected': detect_outliers(X),
        'sensor_range_violations': check_sensor_ranges(X),
        'temporal_consistency': check_temporal_patterns(X) if len(X.shape) > 2 else True
    }
    
    quality_report['overall_quality'] = (
        quality_report['missing_values'] == 0 and
        quality_report['infinite_values'] == 0 and
        quality_report['zero_variance_features'] == 0 and
        quality_report['sensor_range_violations'] == 0
    )
    
    return quality_report
```

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Integration Layer**: v2.0  
**Data Pipeline**: Compatible with shared/processed_data structure  
**Industrial Focus**: Manufacturing IoT Predictive Maintenance Systems
