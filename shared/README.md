# Shared Resources and Utilities

## Overview

The `shared/` directory contains centralized resources, optimized model implementations, and utility functions that are used across all experimental phases of the federated learning research framework. This directory serves as the master repository for processed data, model architectures, and common functionality required by both centralized and federated learning experiments.

## Directory Structure

```
shared/
├── __init__.py                    # Package initialization
├── data/                          # Processed datasets and metadata
│   ├── ai4i2020.csv              # AI4I 2020 Predictive Maintenance Dataset
│   ├── failure_type_map.json     # Failure type categorization mapping
│   └── feature_columns.txt       # Dataset feature descriptions
├── models/                        # Optimized neural network implementations
│   ├── __init__.py               # Model package initialization
│   ├── pdm_models.py            # Legacy model implementations
│   └── step1a_optimized_models.py # Research-grade optimized architectures
├── processed_data/               # Preprocessed data for federated learning
│   ├── metadata.json            # Dataset preprocessing metadata
│   ├── multiclass/              # Multi-class classification datasets
│   ├── sequences/               # Sequential data for temporal models
│   └── tabular/                 # Tabular data for spatial models
└── utils/                       # Utility functions and configurations
    ├── __init__.py              # Utilities package initialization
    ├── step1_config.py          # Experimental configuration management
    ├── step1_config.yaml        # YAML configuration file
    ├── step1_data_preparation.py # Data preprocessing and distribution
    ├── step1_data_utils.py      # Data handling utilities
    ├── step1_experiment_runner.py # Experiment orchestration utilities
    └── temporal_data_preparation.py # Time-series data preprocessing
```

## Core Components

### 1. Data Resources (`data/`)

#### AI4I 2020 Predictive Maintenance Dataset
The foundation dataset for all experimental work, containing industrial IoT sensor measurements for predictive maintenance research.

**Dataset Characteristics**:
- **Source**: Synthetic dataset based on real industrial scenarios
- **Samples**: 10,000 data points representing various operational conditions
- **Features**: 10 numerical sensor measurements
- **Target**: Binary classification (failure/no-failure)
- **Format**: CSV with standardized feature names and preprocessing

**Feature Description**:
1. **Air Temperature [K]**: Ambient air temperature measurements
2. **Process Temperature [K]**: Process-specific temperature readings
3. **Rotational Speed [rpm]**: Motor rotational speed measurements
4. **Torque [Nm]**: Applied torque measurements
5. **Tool Wear [min]**: Cumulative tool wear time
6. **Machine Failure**: Binary target variable (0: no failure, 1: failure)

#### Failure Type Mapping (`failure_type_map.json`)
Comprehensive categorization of failure modes for enhanced analysis:
```json
{
  "TWF": "Tool Wear Failure",
  "HDF": "Heat Dissipation Failure", 
  "PWF": "Power Failure",
  "OSF": "Overstrain Failure",
  "RNF": "Random Failure"
}
```

### 2. Model Implementations (`models/`)

#### Optimized Neural Network Architectures (`step1a_optimized_models.py`)

The research framework implements three state-of-the-art neural network architectures specifically optimized for industrial predictive maintenance:

##### OptimizedCNNModel
**Architecture**: Multi-layer convolutional neural network with adaptive feature extraction

```python
class OptimizedCNNModel(nn.Module):
    """
    Convolutional Neural Network optimized for industrial sensor data analysis.
    
    Architecture:
    - Input reshaping for 1D convolutions
    - Multi-scale feature extraction (filters: 32, 64, 128)
    - Adaptive global pooling
    - Dropout regularization (0.2)
    - Fully connected classification layers
    """
```

**Key Features**:
- **Input Processing**: Handles tabular sensor data through 1D convolutions
- **Feature Extraction**: Multi-scale temporal pattern recognition
- **Regularization**: Dropout and batch normalization for robust training
- **Output**: Binary classification for failure prediction

**Hyperparameters** (Optimized through grid search):
- Hidden Units: 64
- Dropout Rate: 0.2
- Activation: ReLU
- Optimizer: Adam with learning rate 0.01

##### OptimizedLSTMModel
**Architecture**: Bidirectional LSTM with attention mechanism for temporal dependency modeling

```python
class OptimizedLSTMModel(nn.Module):
    """
    Long Short-Term Memory network for sequential pattern recognition.
    
    Architecture:
    - Bidirectional LSTM layers (2 layers, 64 hidden units)
    - Attention mechanism for relevant time step focusing
    - Dropout regularization between layers
    - Dense classification head
    """
```

**Key Features**:
- **Temporal Modeling**: Captures long-term dependencies in sensor sequences
- **Bidirectional Processing**: Forward and backward temporal information integration
- **Attention Mechanism**: Selective focus on critical time steps
- **Memory Efficiency**: Optimized for industrial real-time constraints

**Hyperparameters**:
- Hidden Size: 64
- Number of Layers: 2
- Dropout: 0.2
- Sequence Length: 10

##### OptimizedHybridModel (CNN-LSTM)
**Architecture**: Combined spatial-temporal feature extraction for comprehensive pattern recognition

```python
class OptimizedHybridModel(nn.Module):
    """
    Hybrid CNN-LSTM architecture combining spatial and temporal feature extraction.
    
    Architecture:
    - CNN frontend for spatial feature extraction
    - LSTM backend for temporal sequence modeling
    - Feature fusion layer
    - Multi-task learning capabilities
    """
```

**Key Features**:
- **Multi-Modal Processing**: Simultaneous spatial and temporal pattern analysis
- **Feature Fusion**: Intelligent combination of CNN and LSTM representations
- **Scalability**: Adaptable to various sensor configurations
- **Robustness**: Superior performance on heterogeneous data distributions

**Design Rationale**:
The hybrid architecture addresses the limitations of individual models by combining:
- CNN's strength in spatial pattern recognition
- LSTM's capability in temporal dependency modeling
- Enhanced representation learning through feature fusion

### 3. Processed Data (`processed_data/`)

#### Data Preprocessing Pipeline
The preprocessing system transforms raw AI4I data into federated learning-ready formats:

**Tabular Data Processing**:
- Feature normalization and standardization
- Missing value imputation
- Categorical encoding for failure types
- Train/validation/test splits per datasite

**Sequential Data Processing**:
- Time-series sequence generation (sliding window approach)
- Temporal feature engineering
- Sequence padding and normalization
- Temporal train/validation/test distribution

**Multi-class Data Processing**:
- Extended classification beyond binary failure prediction
- Failure type categorization
- Class balancing and stratification
- Multi-label encoding support

#### Metadata Management (`metadata.json`)
Comprehensive tracking of preprocessing operations:
```json
{
  "dataset_info": {
    "source": "AI4I_2020.csv",
    "total_samples": 10000,
    "feature_count": 10,
    "target_classes": 2
  },
  "preprocessing": {
    "normalization": "MinMaxScaler",
    "train_split": 0.7,
    "validation_split": 0.15,
    "test_split": 0.15
  },
  "federated_splits": {
    "num_datasites": 3,
    "distribution_strategy": "iid",
    "alpha": 0.5
  }
}
```

### 4. Utility Functions (`utils/`)

#### Configuration Management (`step1_config.py`, `step1_config.yaml`)
Centralized configuration system for experimental reproducibility:

**Key Configuration Categories**:
- **Model Parameters**: Architecture-specific hyperparameters
- **Training Parameters**: Learning rates, batch sizes, epochs
- **Federated Learning Parameters**: Aggregation methods, communication rounds
- **Data Distribution Parameters**: IID/Non-IID settings, alpha values

#### Data Preparation Utilities (`step1_data_preparation.py`)
Comprehensive data preprocessing and distribution functions:

**Core Functions**:
```python
def create_federated_datasets(num_clients, distribution='iid', alpha=0.5):
    """Create federated data splits with specified distribution characteristics."""
    
def prepare_temporal_sequences(data, sequence_length=10, overlap=0.5):
    """Generate temporal sequences for LSTM and hybrid models."""
    
def normalize_features(data, method='minmax'):
    """Apply feature normalization with specified method."""
```

#### Experiment Management (`step1_experiment_runner.py`)
Utilities for experiment orchestration and result management:

**Functionality**:
- Experiment configuration validation
- Result collection and aggregation
- Statistical analysis preparation
- Visualization data formatting

## Integration with Experimental Phases

### Phase 0: Data Analysis
- Provides preprocessed datasets for exploratory analysis
- Supplies feature descriptions and metadata
- Enables reproducible data loading and visualization

### Phase 1a: Central Models
- Supplies optimized model architectures for baseline training
- Provides standardized data preprocessing
- Enables fair comparison across model types

### Phase 1b: Federated Learning Simulation
- Provides federated data distribution utilities
- Supplies model implementations for distributed training
- Enables consistent experimental setup across simulation runs

### Phase 1c: Network Federated Learning
- Provides real-time data loading for PySyft datasites
- Supplies production-ready model implementations
- Enables seamless integration with network infrastructure

## Usage Examples

### Loading Optimized Models
```python
from shared.models.step1a_optimized_models import OptimizedCNNModel, OptimizedLSTMModel, OptimizedHybridModel

# Initialize models with standard configuration
cnn_model = OptimizedCNNModel(input_features=10, num_classes=2)
lstm_model = OptimizedLSTMModel(input_features=10, num_classes=2, sequence_length=10)
hybrid_model = OptimizedHybridModel(input_features=10, num_classes=2, sequence_length=10)
```

### Data Preprocessing
```python
from shared.utils.step1_data_preparation import create_federated_datasets
from shared.utils.step1_data_utils import load_ai4i_dataset

# Load and preprocess data
data = load_ai4i_dataset()
federated_datasets = create_federated_datasets(
    data, 
    num_clients=3, 
    distribution='non_iid', 
    alpha=0.5
)
```

### Configuration Management
```python
from shared.utils.step1_config import ExperimentConfig

# Load experimental configuration
config = ExperimentConfig.from_yaml('shared/utils/step1_config.yaml')
print(f"Model type: {config.model_type}")
print(f"Learning rate: {config.learning_rate}")
```

## Quality Assurance

### Model Validation
- **Architecture Testing**: Comprehensive unit tests for model forward passes
- **Performance Benchmarking**: Standardized performance validation
- **Memory Profiling**: Resource usage optimization
- **Gradient Verification**: Backpropagation correctness validation

### Data Integrity
- **Consistency Checks**: Cross-phase data format validation
- **Distribution Verification**: Statistical validation of federated splits
- **Preprocessing Validation**: Input-output transformation verification
- **Metadata Synchronization**: Automatic metadata consistency maintenance

### Reproducibility Standards
- **Seed Management**: Deterministic random number generation
- **Version Control**: Comprehensive change tracking
- **Configuration Validation**: Parameter consistency enforcement
- **Result Verification**: Cross-phase result consistency checks

## Future Enhancements

### Planned Extensions
1. **Additional Datasets**: Integration of other industrial IoT datasets
2. **Model Architecture Expansion**: Transformer and attention-based models
3. **Advanced Preprocessing**: Automated feature engineering and selection
4. **Real-Time Optimization**: Stream processing and online learning support

### Research Integration
1. **Multi-Modal Support**: Vision, audio, and sensor fusion capabilities
2. **Transfer Learning**: Pre-trained model integration
3. **Quantum ML**: Quantum neural network implementations
4. **Explainable AI**: Model interpretability and feature importance analysis

---

**Maintained by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Component Version**: 1.0  
**Integration Level**: Cross-Phase Compatibility
