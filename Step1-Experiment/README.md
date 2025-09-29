# Phase 1: Experimental Framework Implementation

## Overview

Phase 1 represents the core experimental implementation of the federated learning research framework. This phase encompasses three distinct but interconnected experimental approaches: centralized baseline model development, simulated federated learning, and real network federated learning implementation. The phase systematically addresses the research questions through progressively complex experimental scenarios.

## Research Architecture

### Experimental Progression

The phase follows a methodical progression from centralized to fully distributed implementations:

```
Phase 1a: Central Models (Baseline)
    ↓
Phase 1b: Federated Learning Simulation
    ↓  
Phase 1c: Network Federated Learning (Real Implementation)
```

### Research Questions Addressed

1. **RQ1**: Federated Learning Privacy-Performance Trade-offs
2. **RQ2**: FL-EC Integration Impact on Anomaly Detection
3. **RQ3**: Real-time Fault Detection Enhancement
4. **RQ4**: Latency Reduction through Edge Computing
5. **RQ5**: Framework Scalability Assessment

## Directory Structure

```
Step1-Experiment/
├── README.md                     # This comprehensive documentation
├── requirements.txt              # Python dependencies for all experiments
├── central/                      # Phase 1a: Centralized baseline models
│   ├── README.md                # Central models documentation
│   ├── step1a_central_models.ipynb                    # Interactive model development
│   ├── step1a_central_models_bgmode_EXECUTED.ipynb    # Background execution results
│   ├── step1a_central_models_perf_dashboard.html      # Performance dashboard
│   ├── step1a_central_models.py                       # Standalone execution script
│   └── statistical_model_comparison.png               # Statistical analysis results
├── federated/                    # Phase 1b: Simulated federated learning
│   ├── README.md                # Federated learning documentation  
│   ├── step1b_federated_learning_clean_FINAL.ipynb   # Final federated implementation
│   ├── step1b_federated_learning_RESULTS.ipynb       # Results analysis
│   ├── step1b_fl_32rounds_48exp_results_analysis.ipynb # Statistical analysis
│   ├── federated_client.py      # Federated client implementation
│   ├── federated_server.py      # Federated server coordination
│   ├── results/                 # Experimental results and analysis
│   └── logs/                    # Execution logs and debugging information
└── NetworkFed/                  # Phase 1c: Real network federated learning
    ├── README.md                # Network federated learning documentation
    ├── run_enhanced_experiments.py # Main experiment orchestrator
    ├── algorithms/              # Federated learning algorithms
    ├── datasite/               # PySyft datasite infrastructure  
    ├── monitoring/             # Real-time monitoring systems
    ├── config/                 # Configuration management
    ├── utils/                  # Utility functions and helpers
    └── results/                # Experimental results and analysis
```

## Experimental Methodology

### Statistical Framework

All experiments follow rigorous statistical methodology to ensure academic validity:

#### Sample Size Determination
- **Central Models**: 32 independent training runs per model
- **Federated Learning**: 32 independent experimental repetitions
- **Network Implementation**: 32 multi-run experiments for significance testing

#### Statistical Significance Testing
- **ANOVA**: Analysis of variance for multi-group comparisons
- **Post-hoc Tests**: Tukey HSD for pairwise comparisons
- **Effect Size**: Cohen's d and eta-squared calculations
- **Non-parametric Tests**: Friedman test for non-normal distributions

#### Confidence Intervals
- **95% Confidence Level**: Standard for all statistical reporting
- **Bootstrap Methods**: Robust confidence interval estimation
- **Bonferroni Correction**: Multiple comparison error control

### Experimental Controls

#### Randomization
- **Seed Control**: Fixed random seeds for reproducibility
- **Data Shuffling**: Consistent randomization across experiments
- **Initialization**: Standardized model weight initialization

#### Environmental Controls
- **Hardware Consistency**: Standardized computational environment
- **Software Versions**: Fixed dependency versions
- **Resource Allocation**: Consistent memory and CPU allocation

#### Validation Methods
- **Cross-Validation**: K-fold validation within each experimental condition
- **Hold-out Testing**: Independent test sets for unbiased evaluation
- **Temporal Validation**: Time-series aware validation where applicable

## Performance Evaluation Framework

### Metrics Suite

#### Classification Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate (failure detection accuracy)
- **Recall**: Sensitivity to actual failures
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
- **AUC-PR**: Area under precision-recall curve

#### System Performance Metrics
- **Training Time**: Model convergence duration
- **Inference Latency**: Prediction response time
- **Memory Usage**: Peak memory consumption
- **Communication Overhead**: Data transmission costs (federated scenarios)

#### Statistical Metrics
- **Mean Performance**: Average across multiple runs
- **Standard Deviation**: Performance variability measure
- **Confidence Intervals**: Statistical uncertainty quantification
- **Effect Size**: Practical significance assessment

### Evaluation Protocols

#### Train-Validation-Test Split
- **Training**: 70% of data for model optimization
- **Validation**: 15% for hyperparameter tuning and early stopping
- **Testing**: 15% for unbiased final evaluation

#### Cross-Validation Strategy
- **K-Fold**: 5-fold cross-validation for robust performance estimation
- **Stratified**: Maintains class distribution across folds
- **Temporal**: Respects temporal ordering when applicable

#### Statistical Testing Protocol
1. **Normality Testing**: Shapiro-Wilk test for distribution assessment
2. **Homoscedasticity**: Levene's test for variance equality
3. **Parametric vs Non-parametric**: Appropriate test selection
4. **Multiple Comparisons**: Bonferroni or FDR correction

## Quick Start Guide

### Phase 1a: Central Models

```bash
# Navigate to central models directory
cd Step1-Experiment/central/

# Run optimized model experiments (interactive)
jupyter notebook step1a_central_models.ipynb

# Or run standalone script
python step1a_central_models.py --model all --runs 32

# View performance dashboard
open step1a_central_models_perf_dashboard.html
```

### Phase 1b: Federated Learning Simulation

```bash
# Navigate to federated directory
cd Step1-Experiment/federated/

# Run federated learning experiments
jupyter notebook step1b_federated_learning_clean_FINAL.ipynb

# View comprehensive results analysis
jupyter notebook step1b_federated_learning_RESULTS.ipynb

# Run statistical analysis
jupyter notebook step1b_fl_32rounds_48exp_results_analysis.ipynb
```

### Phase 1c: Network Federated Learning

```bash
# Navigate to NetworkFed directory
cd Step1-Experiment/NetworkFed/

# Run complete experimental suite (48 experiments)
python run_enhanced_experiments.py --run-all --max-rounds 10 --runs 32

# Run specific algorithm/model combination
python run_enhanced_experiments.py --algorithm FedAvg --model OptimizedCNN --runs 5

# Run with external datasites (requires configuration)
python run_enhanced_experiments.py --run-all --external-datasites --max-rounds 5

# Monitor experiments in real-time
# Dashboard available at: http://localhost:8889
# Heartbeat monitor at: http://localhost:8888
```

## Integration Architecture

### Phase Dependencies

#### Sequential Dependencies
1. **Phase 0 → Phase 1a**: EDA insights inform baseline model development
2. **Phase 1a → Phase 1b**: Optimized architectures used in federated simulation
3. **Phase 1b → Phase 1c**: Simulation results guide real implementation

#### Cross-Phase Validation
- **Model Consistency**: Same architectures across all phases
- **Hyperparameter Alignment**: Consistent optimization across experiments
- **Data Consistency**: Identical datasets and preprocessing

### Shared Resources Integration

#### Model Architecture Sharing
```python
# Consistent model loading across phases
from shared.models.step1a_optimized_models import (
    OptimizedCNNModel, 
    OptimizedLSTMModel, 
    OptimizedHybridModel
)

# Standardized model initialization
def create_model(model_type, **config):
    if model_type == 'cnn':
        return OptimizedCNNModel(**config)
    elif model_type == 'lstm':
        return OptimizedLSTMModel(**config)
    elif model_type == 'hybrid':
        return OptimizedHybridModel(**config)
```

#### Data Pipeline Consistency
```python
# Shared data preprocessing
from shared.utils.step1_data_preparation import (
    create_federated_datasets,
    normalize_features,
    prepare_temporal_sequences
)

# Consistent preprocessing across phases
def preprocess_data(data, phase_config):
    if phase_config.model_type in ['lstm', 'hybrid']:
        data = prepare_temporal_sequences(data, **phase_config.sequence_params)
    
    data = normalize_features(data, method=phase_config.normalization)
    
    if phase_config.federated:
        data = create_federated_datasets(data, **phase_config.federated_params)
    
    return data
```

## Data Preparation Status

### ✅ Completed Data Processing

The comprehensive data preparation pipeline has been successfully executed:

#### Dataset Specifications
- **Source**: AI4I 2020 Predictive Maintenance Dataset
- **Total Samples**: 10,000 industrial IoT sensor records
- **Features**: 10 engineered features for optimal model performance
- **Classes**: Binary classification (failure/no-failure) + 6-class multiclass

#### Feature Engineering Results
1. **Air temperature [K]** - Original sensor reading
2. **Process temperature [K]** - Original sensor reading  
3. **Rotational speed [rpm]** - Original sensor reading
4. **Torque [Nm]** - Original sensor reading
5. **Tool wear [min]** - Original sensor reading
6. **Temperature_diff** - Engineered: Process temp - Air temp
7. **Power_estimate** - Engineered: Torque × Angular velocity
8. **Tool_wear_normalized** - Engineered: Normalized tool wear values
9. **Torque_speed_ratio** - Engineered: Efficiency indicator
10. **Type_encoded** - Engineered: Label-encoded machine type (L/M/H → 0/1/2)

#### Data Distribution
- **Training Set**: 11,594 samples (SMOTE balanced 1:1 ratio)
- **Validation Set**: 2,000 samples (original distribution)
- **Test Set**: 2,000 samples (original distribution)
- **Sequence Length**: 10 timesteps with 50% overlap (LSTM/Hybrid models)

## Resource Requirements

### Computational Resources

#### Hardware Specifications
- **CPU**: Multi-core processors for parallel computation (minimum 8 cores recommended)
- **Memory**: Minimum 16GB RAM for large-scale experiments (32GB recommended)
- **Storage**: SSD storage for fast I/O (minimum 100GB available space)
- **Network**: High-speed internet for distributed experiments

#### Software Dependencies
```
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine learning frameworks  
torch>=1.9.0
scikit-learn>=1.0.0

# Federated learning
syft>=0.6.0

# Visualization and analysis
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Statistical analysis
statsmodels>=0.12.0
scikit-posthocs>=0.6.0

# Utilities
pyyaml>=5.4.0
tqdm>=4.60.0
jupyter>=1.0.0
```

### Time Requirements

#### Execution Time Estimates
- **Phase 1a (Central)**: ~4-6 hours for complete 32-run analysis
- **Phase 1b (Federated)**: ~8-12 hours for complete 48×32 experiment suite
- **Phase 1c (Network)**: ~12-24 hours for complete network experiments

---

**Experimental Framework**: Comprehensive Multi-Phase Implementation  
**Statistical Rigor**: Academic Research Standards with 32-Run Validation  
**Reproducibility**: Full Documentation and Version Control  
**Scalability**: Support for Large-Scale Federated Learning Experiments

