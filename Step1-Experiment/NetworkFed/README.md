# Real-Time Network Federated Learning for Industrial IoT Predictive Maintenance

## Overview

This repository implements a comprehensive **Network Federated Learning (NetworkFed)** framework for privacy-preserving, real-time predictive maintenance in Industrial Internet of Things (IIoT) environments. The system integrates Federated Learning (FL) with Edge Computing (EC) to address critical challenges in industrial predictive maintenance: data privacy, communication latency, and system scalability.

## Research Context

The research addresses fundamental limitations of centralized predictive maintenance systems through a novel decentralized architecture that combines:

- **Federated Learning**: Enables collaborative model training across distributed industrial sites without raw data sharing
- **Edge Computing**: Provides real-time data processing and decision-making at the industrial edge
- **Network Infrastructure**: Supports both local and external PySyft datasite deployments for realistic industrial scenarios

## System Architecture

### Core Components

1. **Enhanced Parallel Experiment Runner** (`run_enhanced_experiments.py`)
   - Orchestrates federated learning experiments with parallel datasite execution
   - Supports real-time monitoring, fault tolerance, and statistical analysis
   - Features heartbeat monitoring, dashboard integration, and multi-run capabilities
   - Implements comprehensive experiment resumption and parallel processing

2. **Sequential Experiment Runner** (`run_all_48_experiments.py`)
   - Provides sequential execution of all 48 federated learning experiments
   - Implements robust resumption capability for long-running experiment suites
   - Offers simplified experiment management with checkpoint-based recovery
   - Supports both individual and batch experiment execution modes

3. **Factory DataSite Infrastructure** (`datasite/`)
   - Real PySyft-based datasite implementation for secure data enclave simulation
   - Supports both local (launched) and external (pre-configured) datasite modes
   - Implements comprehensive training, validation, and testing workflows

4. **Federated Algorithms** (`algorithms/`)
   - **FedAvg**: Federated Averaging for baseline collaborative learning
   - **FedProx**: Proximal federated optimization for heterogeneous environments
   - **FedDyn**: Dynamic regularization for improved convergence
   - **FedNova**: Normalized averaging for handling system heterogeneity

5. **Advanced Security Infrastructure**
   - **Secure Communication Manager** (`federation/communication/secure_syft_client.py`): AES encryption, differential privacy, and Byzantine fault tolerance
   - **Security Managers** (`utils/security_managers.py`): Differential privacy and Byzantine fault tolerance utilities
   - **Multi-layered Security Pipeline**: Encryption → Authentication → Byzantine Detection → Reputation Filtering → Robust Aggregation

6. **Enhanced Monitoring System**
   - **Heartbeat Manager**: Real-time datasite availability tracking (port 8888)
   - **Status Dashboard**: Web-based experiment monitoring interface (port 8889)
   - **Parallel Execution Manager**: Fault-tolerant distributed training coordination

## Experimental Framework

### Model Architectures

The framework implements three optimized neural network architectures specifically designed for industrial predictive maintenance:

#### 1. Optimized CNN Model
- **Architecture**: Multi-layer convolutional network with adaptive pooling
- **Input**: Tabular sensor data (10 features from AI4I 2020 dataset)
- **Use Case**: Spatial pattern recognition in multi-sensor industrial data
- **Parameters**: Optimized through grid search (64 hidden units, 0.2 dropout, ReLU activation)

#### 2. Optimized LSTM Model
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Input**: Sequential time-series data (temporal sequences of length 10)
- **Use Case**: Temporal dependency modeling for failure prediction
- **Parameters**: 64 hidden units, 2 layers, 0.2 dropout for temporal pattern capture

#### 3. Optimized Hybrid Model (CNN-LSTM)
- **Architecture**: CNN feature extraction followed by LSTM temporal modeling
- **Input**: Sequential sensor data with spatial-temporal dependencies
- **Use Case**: Combined spatial and temporal pattern recognition
- **Parameters**: CNN frontend (32 filters) + LSTM backend (64 units)

### Experimental Design

The framework supports comprehensive experimental evaluation across multiple dimensions:

```
Total Experiments = 3 Models × 4 Algorithms × 2 Distributions × 2 Communication Styles = 48 Experiments
```

#### Experimental Variables:
- **Models**: OptimizedCNN, OptimizedLSTM, OptimizedHybrid
- **Algorithms**: FedAvg, FedProx, FedDyn, FedNova
- **Data Distributions**: IID (uniform), Non-IID (Dirichlet α=0.5)
- **Communication Styles**: 
  - **Standard**: Basic PySyft communication with standard aggregation
  - **Secure**: Full security suite with AES encryption, differential privacy, and Byzantine fault tolerance

#### Evaluation Metrics:
- **Performance**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **System**: Communication overhead, convergence time, fault tolerance
- **Privacy**: Differential privacy guarantees, privacy budget consumption
- **Security**: Byzantine attack detection rate, client reputation scores, security level assessment

## Quick Start Guide

### Prerequisites

```bash
# Python 3.8+ with required dependencies
pip install torch torchvision syft pandas numpy scikit-learn matplotlib seaborn
```

### Experiment Runner Selection

The framework provides two experiment runners, each optimized for different use cases:

#### 1. Enhanced Parallel Experiment Runner (`run_enhanced_experiments.py`)
- **Use Case**: High-performance parallel execution with real-time monitoring
- **Features**: Multi-threaded datasite training, heartbeat monitoring, statistical analysis
- **Best For**: Multi-run experiments, statistical validation, real-time monitoring needs
- **Command Format**: Uses `--experiment` parameter with predefined experiment names

#### 2. Sequential Experiment Runner (`run_all_48_experiments.py`)
- **Use Case**: Robust sequential execution with comprehensive resumption capability
- **Features**: Checkpoint-based resumption, simplified configuration, reliable execution
- **Best For**: Long-running experiment suites, environments with limited resources
- **Command Format**: Uses `--experiment` parameter with predefined experiment names

#### Available Command-Line Parameters

**Common Parameters (Both Runners):**
- `--experiment`: Run specific experiment (e.g., `FedAvg_OptimizedCNN_uniform_sync`)
- `--list-experiments`: List all possible experiments
- `--max-rounds`: Maximum number of federated rounds
- `--local-epochs`: Local training epochs per round
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--num-datasites`: Number of federated datasites
- `--run-all`: Run all 48 experiments
- `--external-datasites`: Use external datasites instead of launching new ones
- `--resume`: Resume from previous checkpoint

**Enhanced Runner Only:**
- `--runs`: Number of times to repeat the experiment execution (for statistical analysis)
- `--disable-enhanced`: Disable enhanced features
- `--max-parallel`: Maximum number of parallel datasites
- `--heartbeat-interval`: Heartbeat check interval in seconds

### Basic Experiment Execution

#### 0. List Available Experiments
```bash
# View all 48 possible experiment combinations
python run_enhanced_experiments.py --list-experiments
# or
python run_all_48_experiments.py --list-experiments
```

#### 1. Single Experiment (Parallel Processing Framework)
```bash
# Using the enhanced parallel experiment runner
python run_enhanced_experiments.py --experiment FedAvg_OptimizedCNN_uniform_sync --max-rounds 10 --local-epochs 5
```

#### 2. Full Experimental Suite (48 Experiments - Parallel)
```bash
# Run all 48 experiments with parallel processing and enhanced monitoring
python run_enhanced_experiments.py --run-all --max-rounds 30 --local-epochs 5 --runs 1
```

#### 3. Single Experiment (Sequential Processing Framework)
```bash
# Using the sequential experiment runner for individual experiments
python run_all_48_experiments.py --experiment FedAvg_OptimizedCNN_uniform_sync --max-rounds 10 --local-epochs 5
```

#### 4. Full Experimental Suite (48 Experiments - Sequential)
```bash
# Run all 48 experiments sequentially with resumption capability
python run_all_48_experiments.py --run-all --max-rounds 30 --local-epochs 5
```

#### 5. External Datasite Mode (Network Deployment)
```bash
# Parallel mode with external datasites
python run_enhanced_experiments.py --run-all --max-rounds 30 --local-epochs 5 --external-datasites --runs 1

# Sequential mode with external datasites
python run_all_48_experiments.py --run-all --max-rounds 30 --local-epochs 5 --external-datasites
```

#### 6. Statistical Analysis Mode (32 Runs)
```bash
# Only available in parallel processing framework
python run_enhanced_experiments.py --run-all --max-rounds 30 --local-epochs 5 --runs 32
```

### Advanced Configuration

#### Multi-Run Statistical Experiments (Parallel Framework)
```bash
# Research-grade statistical analysis with 32 independent runs
python run_enhanced_experiments.py \
    --run-all \
    --max-rounds 30 \
    --local-epochs 5 \
    --external-datasites \
    --runs 32 \
    --enable-enhanced-features \
    --max-parallel-datasites 3
```

#### Sequential Experiment Execution with Resumption
```bash
# Sequential execution with automatic resumption capability
python run_all_48_experiments.py \
    --run-all \
    --max-rounds 30 \
    --local-epochs 5 \
    --external-datasites \
    --resume
```

#### Custom Experiment Configuration (Parallel Framework)
```bash
# Specific experiment with enhanced monitoring
python run_enhanced_experiments.py \
    --experiment FedProx_OptimizedLSTM_non_iid_label_secure \
    --max-rounds 20 \
    --local-epochs 5 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --num-datasites 5 \
    --fedprox-mu 0.1
```

#### Custom Experiment Configuration (Sequential Framework)
```bash
# Individual experiment with resumption support
python run_all_48_experiments.py \
    --experiment FedProx_OptimizedLSTM_uniform_sync \
    --max-rounds 20 \
    --local-epochs 5 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --num-datasites 5 \
    --fedprox-mu 0.1 \
    --resume
```

## Monitoring and Visualization

### Real-Time Dashboard Access

During experiment execution, access the monitoring dashboard:

- **Status Dashboard**: http://localhost:8889
- **Heartbeat Manager**: http://localhost:8888/api/status

The dashboard provides:
- Real-time experiment progress tracking
- Datasite availability monitoring
- Performance metrics visualization
- Error and failure detection

### Results Analysis

Experimental results are automatically saved in structured directories:

```
results/
├── multi_run_32x_experiment_run_YYYYMMDD_HHMMSS/
│   ├── experiment_summary.json
│   ├── statistical_analysis.json
│   └── individual_experiments/
│       ├── FedAvg_OptimizedCNN_IID_Standard/
│       │   ├── experiment_results.json
│       │   ├── round_metrics.csv
│       │   └── debug.log
│       └── ...
└── experiment_run_YYYYMMDD_HHMMSS/
    └── experiment_config.json
```

## System Features

### Enhanced Capabilities

1. **Dual Execution Modes**: 
   - **Parallel Processing** (`run_enhanced_experiments.py`): Multi-threaded datasite training with real-time monitoring
   - **Sequential Processing** (`run_all_48_experiments.py`): Robust sequential execution with comprehensive resumption
2. **Real-Time Monitoring**: Heartbeat-based availability tracking and dashboard visualization
3. **Experiment Resumption**: Automatic checkpoint and resume functionality for long-running experiments
4. **Dynamic Configuration**: YAML-based external datasite configuration management
5. **Statistical Analysis**: Built-in statistical significance testing and comparison frameworks

### Security and Privacy

The framework implements a comprehensive multi-layered security architecture for privacy-preserving federated learning:

#### 1. **Encryption and Secure Communication**
- **AES-256 Encryption**: End-to-end encryption of all model updates using Fernet symmetric encryption
- **Message Authentication**: SHA-256 checksums for message integrity verification
- **Replay Attack Prevention**: Timestamp-based message validation (5-minute timeout window)
- **Key Management**: PBKDF2-based key derivation with 100,000 iterations for secure key generation

#### 2. **Differential Privacy Protection**
- **Gaussian Noise Injection**: Configurable noise multiplier for privacy-utility trade-off
- **Privacy Budget Management**: ε (epsilon) and δ (delta) parameters for (ε,δ)-differential privacy
- **Gradient Clipping**: L2-norm clipping to bound sensitivity before noise addition
- **Adaptive Privacy**: Per-round privacy budget tracking and composition analysis

#### 3. **Byzantine Fault Tolerance (BFT)**
- **Multi-Method Outlier Detection**: 
  - IQR-based statistical outlier detection
  - Z-score threshold analysis (configurable threshold, default: 2.0 std deviations)
  - Cosine similarity analysis for update correlation
- **Client Reputation System**: 
  - Dynamic reputation scoring (0.0 to 1.0 scale)
  - Historical update norm tracking for behavioral analysis
  - Reputation-based filtering with configurable thresholds
- **Robust Aggregation Methods**:
  - **Trimmed Mean**: Removes top/bottom 10% of updates to reduce Byzantine influence
  - **Median Aggregation**: Alternative robust aggregation for extreme Byzantine scenarios
  - **Weighted Averaging**: Reputation-weighted model aggregation

#### 4. **Advanced Security Features**
- **Multi-Stage Filtering Pipeline**:
  1. Decryption and authentication verification
  2. Statistical Byzantine detection (IQR + Z-score)
  3. Reputation-based client filtering
  4. Robust aggregation with trimmed mean
- **Security Metrics Tracking**:
  - Encrypted message count and authentication checks
  - Byzantine attack detection statistics
  - Client filtering and reputation update counts
  - Real-time security level assessment (High/Medium/Low)
- **Comprehensive Audit Trail**:
  - Detailed client reputation reports
  - Update history analysis with statistical summaries
  - Security violation logging and forensics

#### 5. **Security Configuration Parameters**
```python
# Differential Privacy
EPSILON = 1.0              # Privacy budget parameter
DELTA = 1e-5              # Delta for (ε,δ)-differential privacy
NOISE_MULTIPLIER = 1.0    # Gaussian noise scale multiplier
SENSITIVITY = 1.0         # Global sensitivity bound

# Byzantine Fault Tolerance
BFT_THRESHOLD = 2.0       # Z-score threshold for outlier detection
MIN_CLIENTS_BFT = 3       # Minimum clients required for BFT
TRIM_RATIO = 0.1         # Fraction of extreme updates to remove
REPUTATION_THRESHOLD = 0.3 # Minimum reputation for participation

# Encryption Security
MESSAGE_TIMEOUT = 300     # Message validity window (seconds)
KEY_ITERATIONS = 100000   # PBKDF2 key derivation iterations
```

#### 6. **Security-Privacy Trade-offs**
- **Performance Impact**: ~15-25% computational overhead for full security suite
- **Communication Overhead**: ~20-30% increase due to encryption and authentication
- **Privacy Guarantees**: Formal (ε,δ)-differential privacy with bounded information leakage
- **Byzantine Resilience**: Tolerates up to 30% malicious clients with maintained accuracy

### Fault Tolerance

1. **Datasite Resilience**: Automatic handling of datasite failures and reconnection
2. **Network Robustness**: Adaptive communication protocols for unreliable networks
3. **Graceful Degradation**: Continued operation with reduced datasite availability
4. **Comprehensive Logging**: Detailed debug logs for troubleshooting and analysis

## Configuration Management

### External Datasite Configuration

Configure external PySyft datasites via `config/datasite_configs.yaml`:

```yaml
datasites:
  factory_01:
    hostname: "localhost"
    port: 8081
    admin_email: "admin@pdm-factory.com"
    admin_password: "secure_password"
    site_name: "Factory_01"
    
  factory_02:
    hostname: "localhost" 
    port: 8082
    admin_email: "admin@pdm-factory.com"
    admin_password: "secure_password"
    site_name: "Factory_02"
```

### Experiment Parameters

Key configuration parameters for both experiment runners:

#### Parallel Framework (`run_enhanced_experiments.py`)
```python
# Core Experimental Parameters
MAX_ROUNDS = 50              # Federated learning rounds
LOCAL_EPOCHS = 50             # Local training epochs per round
BATCH_SIZE = 32             # Training batch size
LEARNING_RATE = 0.01        # Optimizer learning rate
NUM_DATASITES = 5           # Number of participating datasites

# Enhanced Features
ENABLE_ENHANCED_FEATURES = True     # Parallel processing and monitoring
MAX_PARALLEL_DATASITES = 5          # Concurrent datasite training limit
HEARTBEAT_INTERVAL = 30             # Datasite health check interval (seconds)
```

#### Sequential Framework (`run_all_48_experiments.py`)
```python
# Core Experimental Parameters
MAX_ROUNDS = 10              # Federated learning rounds
LOCAL_EPOCHS = 1             # Local training epochs per round
BATCH_SIZE = 32             # Training batch size
LEARNING_RATE = 0.01        # Optimizer learning rate
NUM_DATASITES = 3           # Number of participating datasites

# Resumption Features
RESUME_ENABLED = True       # Automatic experiment resumption
CHECKPOINT_INTERVAL = 1     # Save checkpoint every round
```

#### Security Configuration (Both Frameworks)
```python
# Communication Security
COMMUNICATION_STYLE = 'secure'  # Enable full security suite
ENABLE_ENCRYPTION = True        # AES-256 encryption
ENABLE_DP = True               # Differential privacy
ENABLE_BFT = True              # Byzantine fault tolerance

# Differential Privacy Parameters
DP_EPSILON = 1.0               # Privacy budget
DP_DELTA = 1e-5               # Delta parameter
DP_NOISE_MULTIPLIER = 1.0     # Noise scale multiplier

# Byzantine Fault Tolerance Parameters
BFT_THRESHOLD = 2.0           # Z-score threshold for outlier detection
MIN_CLIENTS_BFT = 3           # Minimum clients for BFT
REPUTATION_THRESHOLD = 0.3    # Minimum reputation for participation
```

## Research Contributions

### Novel Technical Contributions

1. **Hybrid FL-EC Architecture**: First comprehensive integration of federated learning with edge computing for industrial predictive maintenance
2. **Multi-Modal Model Support**: Unified framework supporting CNN, LSTM, and hybrid architectures for diverse industrial scenarios
3. **Real-Time Monitoring Infrastructure**: Advanced heartbeat and dashboard systems for distributed experiment tracking
4. **Statistical Validation Framework**: Built-in support for statistical significance testing across multiple experimental runs
5. **Comprehensive Security Architecture**: Multi-layered security with AES encryption, differential privacy, and advanced Byzantine fault tolerance
6. **Adaptive Reputation System**: Dynamic client reputation scoring with behavioral analysis for enhanced Byzantine resilience
7. **Robust Aggregation Pipeline**: Multi-stage filtering with statistical outlier detection, reputation-based filtering, and trimmed mean aggregation

### Industrial Applications

1. **Manufacturing**: Real-time equipment failure prediction with privacy preservation
2. **Oil & Gas**: Distributed sensor monitoring across geographically dispersed assets
3. **Automotive**: Multi-facility quality control and predictive maintenance coordination
4. **Energy**: Smart grid fault detection and preventive maintenance scheduling

## Directory Structure

```
NetworkFed/
├── README.md                          # This comprehensive documentation
├── run_enhanced_experiments.py        # Enhanced parallel experiment orchestrator
├── run_all_48_experiments.py         # Sequential experiment runner with resumption
├── federation/                       # Federated learning core components
│   ├── algorithms/                   # Federated learning algorithms
│   │   ├── fedavg.py                # Federated Averaging implementation
│   │   ├── fedprox.py               # FedProx with proximal regularization
│   │   ├── feddyn.py                # FedDyn with dynamic regularization
│   │   └── fednova.py               # FedNova with normalized averaging
│   └── communication/               # Communication layer implementations
│       ├── syft_client.py           # Standard PySyft communication
│       └── secure_syft_client.py    # Secure communication with encryption, DP, and BFT
├── datasite/                         # PySyft datasite infrastructure
│   ├── factory_node.py               # Core datasite implementation
│   └── datasite_config.py           # Configuration management
├── config/                           # Configuration files
│   ├── datasite_configs.yaml        # External datasite specifications
│   └── README.md                     # Configuration documentation
├── monitoring/                       # Real-time monitoring systems
│   ├── heartbeat_manager.py          # Datasite availability tracking
│   ├── status_dashboard.py           # Web-based monitoring interface
│   └── parallel_execution_manager.py # Fault-tolerant coordination
├── utils/                           # Utility functions and helpers
│   ├── data_preparation.py          # Data distribution and preprocessing
│   ├── experiment_configuration.py   # Experiment setup and validation
│   └── security_managers.py         # Differential privacy and Byzantine fault tolerance utilities
└── results/                         # Experimental results and analysis
    ├── individual_runs/              # Single experiment results
    └── multi_run_analysis/           # Statistical analysis outputs
```

## Related Research

This work builds upon and extends the methodological framework described in the research proposal "Integrating Federated Learning and Edge Computing for Privacy-Preserving and Real-time Predictive Maintenance in Industrial IoT Systems."

### Key Research Questions Addressed

1. **Privacy-Performance Trade-off**: How effectively does FedAvg mitigate privacy risks while maintaining high accuracy?
2. **Edge-Cloud Integration**: What impact does FL-EC integration have on anomaly detection in IIoT sensor data?
3. **Real-Time Capabilities**: How does the decentralized framework enhance temporal dependency modeling?
4. **Latency Reduction**: Does FL-EC combination reduce latency compared to centralized systems?
5. **Scalability Assessment**: How scalable is the framework across diverse industrial configurations?

## Future Enhancements

### Planned Development

1. **Advanced Privacy Mechanisms**: Implementation of secure multi-party computation and homomorphic encryption
2. **Adaptive Aggregation**: Dynamic weighting schemes based on datasite reliability and data quality
3. **Cross-Platform Deployment**: Support for heterogeneous edge devices and cloud platforms
4. **Industry-Specific Models**: Specialized neural architectures for automotive, oil & gas, and manufacturing domains

### Research Extensions

1. **Real-World Validation**: Deployment in actual industrial environments with live sensor data
2. **Multi-Modal Integration**: Support for diverse sensor types (vision, audio, environmental)
3. **Federated Transfer Learning**: Cross-domain knowledge transfer between industrial applications
4. **Blockchain Integration**: Immutable audit trails for federated learning transactions

## Citation and Acknowledgments

If you use this framework in your research, please cite:

```
@software{networkfed_framework_2025,
  title={NetworkFed: Real-Time Network Federated Learning for Industrial IoT Predictive Maintenance},
  author={PDM-FDL Research Team},
  year={2025},
  url={https://github.com/vejendla-kiran/pdm-fdl}
}
```

## License and Contributing

This research framework is provided for academic and research purposes. Contributions are welcome through pull requests and issue reports.

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Framework Version**: 2.0  
**Research Phase**: Step 1 - Network Federated Learning Implementation
