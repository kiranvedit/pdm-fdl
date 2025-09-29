# NetworkFed Models

This directory contains the optimized deep learning models for predictive maintenance in industrial IoT systems. These models are specifically designed and tuned for federated learning deployment across distributed manufacturing environments, incorporating the best hyperparameters discovered through comprehensive hyperparameter optimization in Step 1A.

## Architecture Overview

The models module provides a factory-based architecture for creating, managing, and deploying optimized machine learning models for federated learning scenarios:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NetworkFed Models Architecture              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ModelFactory                             │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐ │   │
│  │  │ CNN Factory   │ │ LSTM Factory  │ │ Hybrid Factory    │ │   │
│  │  │ - 93.95% Acc  │ │ - 94.15% Acc  │ │ - 92.20% Acc      │ │   │
│  │  │ - Optimized   │ │ - Bidirection │ │ - CNN + LSTM      │ │   │
│  │  │ - Conv Layers │ │ - Temporal    │ │ - Spatial+Temporal│ │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  ModelWrapper                               │   │
│  │  - ModelInterface Compliance                               │   │
│  │  - Parameter Get/Set Methods                               │   │
│  │  - Forward Pass Delegation                                 │   │
│  │  - Federated Learning Compatible                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  ModelManager                               │   │
│  │  - Model Lifecycle Management                              │   │
│  │  - State Persistence & Restoration                         │   │
│  │  - Round-based History Tracking                            │   │
│  │  - Multi-model Registration                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Optimized Model Zoo                          │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │   │
│  │  │ OptimizedCNN    │ │ OptimizedLSTM   │ │OptimizedHybrid│ │   │
│  │  │ - 3 Conv Layers │ │ - Bidirectional │ │- CNN Feature  │ │   │
│  │  │ - BatchNorm     │ │ - 64 Hidden     │ │  Extraction   │ │   │
│  │  │ - Dropout 0.3   │ │ - Dropout 0.2   │ │- LSTM Temp    │ │   │
│  │  │ - 32/64/128     │ │ - 1 Layer       │ │  Processing   │ │   │
│  │  │   Filters       │ │ - FC Layers     │ │- Dropout 0.4  │ │   │
│  │  │ - FC 256/128    │ │ - Adam Opt      │ │- 32/64 Filters│ │   │
│  │  └─────────────────┘ └─────────────────┘ └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Components Overview

### **Module Initialization (`__init__.py`)**
```python
from .model_factory import ModelFactory, ModelManager, ModelWrapper
from .optimized_models import OptimizedCNNModel, OptimizedLSTMModel, OptimizedHybridModel

__all__ = [
    'ModelFactory',
    'ModelManager', 
    'ModelWrapper',
    'OptimizedCNNModel',
    'OptimizedLSTMModel',
    'OptimizedHybridModel'
]
```

**Exported Components:**
- **ModelFactory**: Factory pattern for creating optimized model instances
- **ModelManager**: Lifecycle management for multiple models
- **ModelWrapper**: Interface adapter for federated learning compatibility
- **Optimized Model Classes**: Pre-tuned model implementations

---

## 2. Model Factory Pattern (`ModelFactory`)

### **What is the ModelFactory?**

The `ModelFactory` provides a centralized factory pattern for creating optimized model instances with pre-configured hyperparameters discovered through extensive hyperparameter tuning. It ensures consistent model creation across federated learning deployments.

### **Core Factory Features**

#### **Model Creation with Optimized Hyperparameters**
```python
from models import ModelFactory
from core.enums import ModelType

# Create optimized CNN model
cnn_model = ModelFactory.create_model(
    model_type=ModelType.CNN,
    input_dim=10,      # Industrial sensor features
    num_classes=2      # Binary failure prediction
)

# Create optimized LSTM model  
lstm_model = ModelFactory.create_model(
    model_type=ModelType.LSTM,
    input_dim=10,
    num_classes=2
)

# Create optimized Hybrid CNN-LSTM model
hybrid_model = ModelFactory.create_model(
    model_type=ModelType.HYBRID,
    input_dim=10,
    num_classes=2
)
```

**Factory Benefits:**
- **Pre-optimized Hyperparameters**: Models created with best-performing configurations
- **Type Safety**: Enum-based model type specification
- **Error Handling**: Comprehensive error handling for unsupported model types
- **Consistent Interface**: All models wrapped with ModelInterface compliance

#### **Default Configuration Management**
```python
# Get default configurations for each model type
cnn_config = ModelFactory.get_default_config(ModelType.CNN)
lstm_config = ModelFactory.get_default_config(ModelType.LSTM)
hybrid_config = ModelFactory.get_default_config(ModelType.HYBRID)

# Example CNN configuration
cnn_default = {
    'input_dim': 10,
    'num_classes': 2,
    'conv_filters': [32, 64, 128],      # Optimized filter progression
    'fc_hidden': [256, 128],            # Optimized FC layer sizes
    'dropout_rate': 0.3                 # Optimized dropout rate
}

# Example LSTM configuration
lstm_default = {
    'input_dim': 10,
    'num_classes': 2,
    'hidden_size': 128,                 # Optimized hidden size
    'num_layers': 2,                    # Optimized layer count
    'dropout_rate': 0.2,                # Optimized dropout rate
    'bidirectional': True               # Optimized bidirectionality
}
```

### **Model Registry and Extension**

#### **Register Custom Models**
```python
# Define custom optimized model
class CustomIndustrialModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Custom implementation for specific industrial use case
        pass

# Register custom model type
from core.enums import ModelType
custom_type = ModelType.CUSTOM  # Assuming extended enum

ModelFactory.register_model(custom_type, CustomIndustrialModel)

# List all available models
available_models = ModelFactory.list_available_models()
print(f"Available models: {[m.value for m in available_models]}")
```

---

## 3. Model Interface Wrapper (`ModelWrapper`)

### **What is the ModelWrapper?**

The `ModelWrapper` class adapts PyTorch models to conform to the federated learning `ModelInterface`, providing standardized methods for parameter management and forward passes required for distributed training coordination.

### **Interface Compliance Features**

#### **Parameter Management for Federated Learning**
```python
from models import ModelWrapper
import torch.nn as nn

# Wrap any PyTorch model
base_model = OptimizedCNNModel(input_dim=10, num_classes=2)
federated_model = ModelWrapper(base_model)

# Extract parameters for federated aggregation
model_parameters = federated_model.get_parameters()

# Example parameter structure
{
    'conv1.weight': tensor([[...], [...]]),
    'conv1.bias': tensor([...]),
    'bn1.weight': tensor([...]),
    'bn1.bias': tensor([...]),
    'fc1.weight': tensor([[...], [...]]),
    'fc1.bias': tensor([...]),
    # ... all model parameters as tensors
}

# Set parameters from federated aggregation
new_parameters = aggregate_federated_parameters(client_updates)
federated_model.set_parameters(new_parameters)
```

**Parameter Management Features:**
- **Clone & Detach**: Safe parameter extraction without gradient tracking
- **State Dict Compatibility**: Seamless integration with PyTorch state_dict
- **Type Safety**: Consistent tensor type handling across devices
- **Memory Efficiency**: Minimal memory overhead during parameter operations

#### **Forward Pass Delegation**
```python
# Standard forward pass through wrapped model
input_data = torch.randn(32, 10)  # Batch of industrial sensor data
predictions = federated_model.forward(input_data)

# All other PyTorch model methods available through delegation
federated_model.train()                    # Switch to training mode
federated_model.eval()                     # Switch to evaluation mode
federated_model.to(device)                 # Move to GPU/CPU
federated_model.named_parameters()         # Access parameter iterator
```

---

## 4. Model Lifecycle Management (`ModelManager`)

### **What is the ModelManager?**

The `ModelManager` provides comprehensive lifecycle management for multiple models across federated learning rounds, including model registration, state persistence, history tracking, and restoration capabilities.

### **Multi-Model Management**

#### **Model Registration and Retrieval**
```python
from models import ModelManager
from core.enums import ModelType

# Initialize model manager
manager = ModelManager()

# Create and register multiple models for different factory sites
automotive_model = manager.create_and_register_model(
    model_id="automotive_detroit_cnn",
    model_type=ModelType.CNN,
    input_dim=10,
    num_classes=2
)

chemical_model = manager.create_and_register_model(
    model_id="chemical_houston_lstm", 
    model_type=ModelType.LSTM,
    input_dim=10,
    num_classes=2
)

aerospace_model = manager.create_and_register_model(
    model_id="aerospace_seattle_hybrid",
    model_type=ModelType.HYBRID,
    input_dim=10,
    num_classes=2
)

# Retrieve models by ID
automotive = manager.get_model("automotive_detroit_cnn")
chemical = manager.get_model("chemical_houston_lstm")

# List all registered models
all_models = manager.list_models()
print(f"Registered models: {all_models}")
```

### **State Persistence and History**

#### **Round-based State Management**
```python
def federated_training_with_history(manager, model_id, training_rounds):
    """Federated training with comprehensive state history."""
    
    for round_num in range(training_rounds):
        print(f"🔄 Starting federated round {round_num + 1}")
        
        # Save state before training
        manager.save_model_state(model_id, round_num)
        
        # Get current model for training
        current_model = manager.get_model(model_id)
        
        # Simulate federated training round
        # (actual federated training would happen here)
        
        # Example: Update model parameters
        updated_params = simulate_federated_aggregation(current_model)
        current_model.set_parameters(updated_params)
        
        print(f"✅ Completed round {round_num + 1}, state saved")
    
    # Save final state
    manager.save_model_state(model_id, training_rounds)

# Execute training with history
federated_training_with_history(manager, "automotive_detroit_cnn", 10)
```

#### **State Restoration and Rollback**
```python
# Restore model to specific round
def restore_model_checkpoint(manager, model_id, target_round):
    """Restore model to a specific training round."""
    
    try:
        # Restore to specific round
        manager.restore_model_state(model_id, round_num=target_round)
        print(f"✅ Model {model_id} restored to round {target_round}")
        
        # Verify restoration
        restored_model = manager.get_model(model_id)
        return restored_model
        
    except Exception as e:
        print(f"❌ Failed to restore model: {e}")
        return None

# Restore to round 5
restored_model = restore_model_checkpoint(manager, "automotive_detroit_cnn", 5)

# Restore to latest state (most recent)
manager.restore_model_state("chemical_houston_lstm")  # No round_num = latest
```

### **Model Cleanup and Resource Management**
```python
# Remove specific model and its history
manager.remove_model("automotive_detroit_cnn")

# Cleanup models with low performance
def cleanup_underperforming_models(manager, performance_threshold=0.85):
    """Remove models that don't meet performance criteria."""
    
    models_to_remove = []
    
    for model_id in manager.list_models():
        # Evaluate model performance (simplified example)
        model = manager.get_model(model_id)
        performance = evaluate_model_performance(model)
        
        if performance < performance_threshold:
            models_to_remove.append(model_id)
    
    for model_id in models_to_remove:
        manager.remove_model(model_id)
        print(f"🗑️ Removed underperforming model: {model_id}")

# Execute cleanup
cleanup_underperforming_models(manager)
```

---

## 5. Optimized Model Architectures

### **OptimizedCNNModel - Convolutional Neural Network**

#### **Model Architecture and Performance**
```python
from models import OptimizedCNNModel

# Create optimized CNN with best hyperparameters
cnn_model = OptimizedCNNModel(input_dim=10, num_classes=2)

# Model architecture details
architecture_summary = {
    'model_type': 'Convolutional Neural Network',
    'accuracy_achieved': '93.95%',
    'optimization_method': 'Hyperparameter Tuning (Step 1A)',
    'conv_layers': 3,
    'conv_filters': [32, 64, 128],
    'fc_layers': 3,
    'fc_hidden': [256, 128],
    'dropout_rate': 0.3,
    'batch_normalization': True,
    'activation': 'ReLU',
    'pooling': 'MaxPool1d'
}
```

**CNN Architecture Breakdown:**
- **Layer 1**: Conv1d(1→32) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
- **Layer 2**: Conv1d(32→64) + BatchNorm + ReLU + MaxPool + Dropout(0.3)  
- **Layer 3**: Conv1d(64→128) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
- **Flatten**: Adaptive flattening based on input dimensions
- **FC1**: Linear(conv_output→256) + ReLU + Dropout(0.3)
- **FC2**: Linear(256→128) + ReLU + Dropout(0.3)
- **Output**: Linear(128→num_classes)

#### **Optimal Training Configuration**
```python
# Best hyperparameters for CNN
cnn_training_config = {
    'batch_size': 32,           # Optimal batch size
    'learning_rate': 0.0005,    # Optimal learning rate
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'step_size': 10,
    'gamma': 0.5,
    'weight_decay': 1e-4,
    'epochs': 50               # Recommended training epochs
}

# Apply configuration during training
optimizer = torch.optim.Adam(
    cnn_model.parameters(), 
    lr=cnn_training_config['learning_rate'],
    weight_decay=cnn_training_config['weight_decay']
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cnn_training_config['step_size'],
    gamma=cnn_training_config['gamma']
)
```

### **OptimizedLSTMModel - Long Short-Term Memory**

#### **Model Architecture and Performance**
```python
from models import OptimizedLSTMModel

# Create optimized LSTM for temporal pattern recognition
lstm_model = OptimizedLSTMModel(
    input_dim=10, 
    num_classes=2, 
    sequence_length=10
)

# Model architecture details
lstm_architecture = {
    'model_type': 'Bidirectional LSTM',
    'accuracy_achieved': '94.15%',  # Highest performing model
    'temporal_processing': True,
    'hidden_size': 64,              # Optimal hidden dimension
    'num_layers': 1,                # Optimal layer count
    'bidirectional': True,          # Better temporal understanding
    'dropout_rate': 0.2,            # Optimal regularization
    'sequence_processing': 'Last output concatenation'
}
```

**LSTM Architecture Breakdown:**
- **LSTM Layer**: Bidirectional LSTM(input_dim→64, layers=1)
- **Hidden State Processing**: Concatenate final forward + backward states
- **FC1**: Linear(128→128) + ReLU + Dropout(0.2)  # 128 = 64*2 (bidirectional)
- **Output**: Linear(128→num_classes)

#### **Temporal Data Handling**
```python
# Example temporal data processing
def process_temporal_data(lstm_model, sequence_data):
    """Process temporal sensor sequences for predictive maintenance."""
    
    # Input shape: (batch_size, sequence_length, features)
    # Example: (32, 10, 10) = 32 batches, 10 time steps, 10 sensor features
    
    # LSTM processes full sequences
    lstm_output = lstm_model(sequence_data)
    
    # Output: (batch_size, num_classes) = (32, 2)
    # Predictions for failure/no-failure classification
    
    return lstm_output

# Optimal training configuration
lstm_training_config = {
    'batch_size': 16,           # Optimal for LSTM memory requirements
    'learning_rate': 0.0005,    # Optimal learning rate
    'optimizer': 'Adam',
    'sequence_length': 10,      # Optimal temporal window
    'gradient_clipping': 1.0    # Prevent gradient explosion
}
```

### **OptimizedHybridModel - CNN-LSTM Fusion**

#### **Model Architecture and Performance**
```python
from models import OptimizedHybridModel

# Create hybrid model combining spatial and temporal processing
hybrid_model = OptimizedHybridModel(
    input_dim=10,
    num_classes=2,
    sequence_length=10
)

# Model architecture details
hybrid_architecture = {
    'model_type': 'Hybrid CNN-LSTM',
    'accuracy_achieved': '92.20%',
    'processing_stages': ['Spatial Feature Extraction', 'Temporal Processing'],
    'cnn_filters': [32, 64],        # Optimal CNN progression
    'lstm_hidden': 128,             # Optimal LSTM hidden size
    'dropout_rate': 0.4,            # Higher regularization for complexity
    'fusion_strategy': 'Sequential CNN→LSTM'
}
```

**Hybrid Architecture Breakdown:**
- **CNN Stage**: 
  - Conv1d(input_dim→32) + BatchNorm + ReLU + MaxPool + Dropout(0.4)
  - Conv1d(32→64) + BatchNorm + ReLU + MaxPool + Dropout(0.4)
- **Reshape**: Convert CNN output for LSTM input
- **LSTM Stage**: LSTM(64→128, layers=1, unidirectional)
- **Classification**: 
  - Linear(128→64) + ReLU + Dropout(0.4)
  - Linear(64→num_classes)

#### **Multi-Modal Data Processing**
```python
def hybrid_data_processing(hybrid_model, industrial_data):
    """Process both tabular and sequence data with hybrid model."""
    
    # For tabular data: (batch_size, features) → (batch_size, features, 1)
    # For sequence data: (batch_size, seq_len, features) → permuted for CNN
    
    # Hybrid model handles both automatically
    predictions = hybrid_model(industrial_data)
    
    return predictions

# Optimal training configuration
hybrid_training_config = {
    'batch_size': 16,           # Optimal for memory efficiency
    'learning_rate': 0.001,     # Higher learning rate for complexity
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'dropout_rate': 0.4,        # Higher regularization
    'gradient_clipping': 1.0    # Important for CNN-LSTM combination
}
```

---

## 6. Factory Function and Utilities

### **Simplified Model Creation**

#### **Factory Function for Quick Deployment**
```python
from models.optimized_models import create_optimized_model

# Quick model creation with optimized hyperparameters
def deploy_models_for_factory(factory_type, data_characteristics):
    """Deploy appropriate models based on factory requirements."""
    
    models = {}
    
    if data_characteristics['has_temporal_patterns']:
        # Use LSTM for temporal data
        models['temporal'] = create_optimized_model(
            model_type='lstm',
            input_dim=data_characteristics['feature_count'],
            num_classes=data_characteristics['num_classes'],
            sequence_length=data_characteristics['sequence_length']
        )
    
    if data_characteristics['has_spatial_patterns']:
        # Use CNN for spatial/feature patterns
        models['spatial'] = create_optimized_model(
            model_type='cnn',
            input_dim=data_characteristics['feature_count'],
            num_classes=data_characteristics['num_classes']
        )
    
    if data_characteristics['complex_patterns']:
        # Use Hybrid for complex multi-modal data
        models['hybrid'] = create_optimized_model(
            model_type='hybrid',
            input_dim=data_characteristics['feature_count'],
            num_classes=data_characteristics['num_classes'],
            sequence_length=data_characteristics['sequence_length']
        )
    
    return models

# Example factory deployment
automotive_data_chars = {
    'feature_count': 10,
    'num_classes': 2,
    'sequence_length': 10,
    'has_temporal_patterns': True,
    'has_spatial_patterns': True,
    'complex_patterns': True
}

automotive_models = deploy_models_for_factory('automotive', automotive_data_chars)
```

### **Training Configuration Management**

#### **Optimized Training Configurations**
```python
from models.optimized_models import OPTIMIZED_TRAINING_CONFIG, BEST_HYPERPARAMETERS

# Access pre-optimized training configurations
def setup_federated_training(model_type):
    """Setup training configuration for federated learning."""
    
    config = OPTIMIZED_TRAINING_CONFIG[model_type].copy()
    hyperparams = BEST_HYPERPARAMETERS[model_type].copy()
    
    # Combine configurations
    training_setup = {
        **config,
        **hyperparams,
        'federated_rounds': 10,
        'local_epochs': 3,
        'aggregation_method': 'FedAvg'
    }
    
    return training_setup

# Setup training for each model type
cnn_federated_config = setup_federated_training('cnn')
lstm_federated_config = setup_federated_training('lstm')
hybrid_federated_config = setup_federated_training('hybrid')

# Example CNN federated configuration
{
    'batch_size': 32,
    'learning_rate': 0.0005,
    'optimizer': 'Adam',
    'conv_filters': [32, 64, 128],
    'fc_hidden': [256, 128],
    'dropout_rate': 0.3,
    'federated_rounds': 10,
    'local_epochs': 3,
    'aggregation_method': 'FedAvg'
}
```

---

## 7. Integration with Federated Learning

### **Model Deployment for Factory Networks**

#### **Multi-Factory Model Distribution**
```python
def setup_federated_models_for_network(factory_configs):
    """Setup models for multi-factory federated learning network."""
    
    factory_models = {}
    
    for factory_id, config in factory_configs.items():
        # Initialize model manager for each factory
        manager = ModelManager()
        
        # Create models based on factory's data characteristics
        factory_models[factory_id] = {}
        
        if config['prefers_cnn']:
            factory_models[factory_id]['cnn'] = manager.create_and_register_model(
                model_id=f"{factory_id}_cnn",
                model_type=ModelType.CNN,
                input_dim=config['input_dim'],
                num_classes=config['num_classes']
            )
        
        if config['prefers_lstm']:
            factory_models[factory_id]['lstm'] = manager.create_and_register_model(
                model_id=f"{factory_id}_lstm", 
                model_type=ModelType.LSTM,
                input_dim=config['input_dim'],
                num_classes=config['num_classes']
            )
        
        if config['prefers_hybrid']:
            factory_models[factory_id]['hybrid'] = manager.create_and_register_model(
                model_id=f"{factory_id}_hybrid",
                model_type=ModelType.HYBRID,
                input_dim=config['input_dim'],
                num_classes=config['num_classes']
            )
    
    return factory_models

# Example factory network configuration
factory_network_config = {
    'automotive_detroit': {
        'input_dim': 10,
        'num_classes': 2,
        'prefers_cnn': True,
        'prefers_lstm': False,
        'prefers_hybrid': True
    },
    'chemical_houston': {
        'input_dim': 10,
        'num_classes': 2,
        'prefers_cnn': False,
        'prefers_lstm': True,
        'prefers_hybrid': False
    },
    'aerospace_seattle': {
        'input_dim': 10,
        'num_classes': 2,
        'prefers_cnn': True,
        'prefers_lstm': True,
        'prefers_hybrid': True
    }
}

# Deploy models across factory network
network_models = setup_federated_models_for_network(factory_network_config)
```

### **Model Performance Comparison**

#### **Performance Metrics Summary**
```python
# Model performance summary for selection guidance
MODEL_PERFORMANCE_GUIDE = {
    'cnn': {
        'best_for': ['Spatial patterns', 'Feature correlations', 'Tabular data'],
        'accuracy_range': '92-94%',
        'training_speed': 'Fast',
        'memory_usage': 'Medium',
        'federated_efficiency': 'High'
    },
    'lstm': {
        'best_for': ['Temporal patterns', 'Sequential data', 'Time series'],
        'accuracy_range': '93-95%',  # Highest performing
        'training_speed': 'Medium',
        'memory_usage': 'High',
        'federated_efficiency': 'Medium'
    },
    'hybrid': {
        'best_for': ['Complex patterns', 'Multi-modal data', 'Comprehensive analysis'],
        'accuracy_range': '91-93%',
        'training_speed': 'Slow',
        'memory_usage': 'High',
        'federated_efficiency': 'Medium'
    }
}

def recommend_model_for_usecase(data_characteristics, performance_requirements):
    """Recommend optimal model based on data and performance requirements."""
    
    recommendations = []
    
    # Performance-based recommendations
    if performance_requirements.get('max_accuracy', False):
        recommendations.append(('lstm', 'Highest accuracy: 94.15%'))
    
    if performance_requirements.get('fastest_training', False):
        recommendations.append(('cnn', 'Fastest training and high federated efficiency'))
    
    if performance_requirements.get('comprehensive_analysis', False):
        recommendations.append(('hybrid', 'Best for complex multi-modal analysis'))
    
    # Data-based recommendations
    if data_characteristics.get('temporal_dominant', False):
        recommendations.append(('lstm', 'Optimal for temporal pattern recognition'))
    
    if data_characteristics.get('spatial_dominant', False):
        recommendations.append(('cnn', 'Optimal for spatial/feature pattern recognition'))
    
    return recommendations
```

---

## 8. Best Practices and Guidelines

### **Model Selection Guidelines**

#### **Choosing the Right Model**
```python
def model_selection_guide():
    """Guidelines for selecting appropriate models."""
    
    selection_criteria = {
        'data_type_considerations': {
            'tabular_data': 'CNN - Excellent for feature correlations',
            'time_series': 'LSTM - Designed for temporal dependencies', 
            'mixed_data': 'Hybrid - Handles both spatial and temporal'
        },
        'performance_priorities': {
            'maximum_accuracy': 'LSTM (94.15% accuracy)',
            'training_efficiency': 'CNN (Fast convergence)',
            'comprehensive_analysis': 'Hybrid (Multi-modal processing)'
        },
        'federated_considerations': {
            'communication_efficiency': 'CNN (Smaller parameter count)',
            'memory_constraints': 'CNN (Lower memory usage)',
            'robustness': 'LSTM (Stable convergence)'
        }
    }
    
    return selection_criteria

# Usage example
selection_guide = model_selection_guide()
```

### **Training Best Practices**

#### **Federated Training Optimization**
```python
def federated_training_best_practices():
    """Best practices for federated training with optimized models."""
    
    practices = {
        'batch_size_optimization': {
            'cnn': 'Use batch_size=32 for optimal convergence',
            'lstm': 'Use batch_size=16 to manage memory',
            'hybrid': 'Use batch_size=16 for complex processing'
        },
        'learning_rate_scheduling': {
            'initial_lr': 'Use model-specific optimized rates',
            'scheduler': 'StepLR with step_size=10, gamma=0.5',
            'adaptation': 'Monitor convergence and adjust'
        },
        'regularization_strategies': {
            'dropout': 'Model-specific optimized rates (0.2-0.4)',
            'weight_decay': '1e-4 for all models',
            'gradient_clipping': '1.0 for LSTM and Hybrid models'
        },
        'federated_specific': {
            'local_epochs': '3-5 epochs per round',
            'aggregation_frequency': 'Every 1-2 rounds',
            'client_sampling': 'Minimum 3 clients per round'
        }
    }
    
    return practices
```

---

## 9. Monitoring and Diagnostics

### **Model Performance Monitoring**

#### **Training Progress Tracking**
```python
class ModelPerformanceMonitor:
    """Monitor model performance during federated training."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.performance_history = {}
    
    def track_round_performance(self, model_id, round_num, metrics):
        """Track performance metrics for each training round."""
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        round_metrics = {
            'round': round_num,
            'accuracy': metrics.get('accuracy', 0.0),
            'loss': metrics.get('loss', float('inf')),
            'convergence_rate': metrics.get('convergence_rate', 0.0),
            'parameter_norm': metrics.get('parameter_norm', 0.0)
        }
        
        self.performance_history[model_id].append(round_metrics)
    
    def detect_training_issues(self, model_id):
        """Detect potential training issues."""
        
        if model_id not in self.performance_history:
            return {'status': 'no_data'}
        
        history = self.performance_history[model_id][-5:]  # Last 5 rounds
        
        issues = []
        
        # Check for convergence stagnation
        if len(history) >= 3:
            recent_accuracies = [h['accuracy'] for h in history[-3:]]
            if max(recent_accuracies) - min(recent_accuracies) < 0.01:
                issues.append('accuracy_stagnation')
        
        # Check for loss explosion
        recent_losses = [h['loss'] for h in history]
        if any(loss > 10.0 for loss in recent_losses):
            issues.append('loss_explosion')
        
        # Check for parameter instability
        recent_norms = [h['parameter_norm'] for h in history]
        if len(recent_norms) >= 2:
            if recent_norms[-1] > recent_norms[-2] * 2:
                issues.append('parameter_instability')
        
        return {'status': 'issues_detected' if issues else 'healthy', 'issues': issues}

# Usage example
monitor = ModelPerformanceMonitor(model_manager)
```

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Model Optimization**: Based on Step 1A Hyperparameter Tuning Results  
**Accuracy Range**: 92.20% - 94.15%  
**Deployment Ready**: Federated Learning in Industrial Manufacturing Networks
