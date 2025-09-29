# Privacy-Preserving Real-Time Predictive Maintenance in Industrial IoT Systems

## Project Overview

This repository implements a comprehensive research framework for **Privacy-Preserving Real-Time Predictive Maintenance in Industrial Internet of Things (IIoT) Systems** through the integration of Federated Learning (FL) and Edge Computing (EC). The research addresses critical challenges in industrial predictive maintenance: data privacy preservation, communication latency reduction, and system scalability enhancement.

## Research Methodology

The research follows a systematic three-phase approach to develop and validate a novel decentralized predictive maintenance framework:

### Phase 0: Data Analysis and Exploration
- **Objective**: Comprehensive analysis of the AI4I 2020 Predictive Maintenance Dataset
- **Location**: `Step0-DataAnalysis/`
- **Outcomes**: Dataset characterization, feature engineering insights, and baseline understanding

### Phase 1: Centralized Model Development and Federated Learning Implementation
- **Objective**: Develop optimized neural network architectures and implement federated learning algorithms
- **Location**: `Step1-Experiment/`
- **Components**:
  - **Central Models**: Baseline centralized training with statistical validation
  - **Federated Learning**: Simulated federated learning with comprehensive algorithm comparison
  - **Network Federated Learning**: Real PySyft-based distributed implementation

## System Architecture

### Core Technologies

1. **Federated Learning Framework**
   - Enables collaborative model training across distributed industrial sites
   - Preserves data privacy by keeping raw data at source locations
   - Implements four state-of-the-art aggregation algorithms

2. **Edge Computing Integration**
   - Provides real-time data processing capabilities at industrial edge devices
   - Reduces communication latency through local computation
   - Supports both local and external datasite configurations

3. **Privacy-Preserving Mechanisms**
   - Differential privacy implementation for enhanced data protection
   - Secure aggregation protocols for model update protection
   - GDPR-compliant data handling procedures

## Directory Structure

```
pdm-fdl/
├── README.md                           # This comprehensive documentation
├── methodology_extracted.txt           # Research proposal and theoretical background
├── shared/                            # Shared utilities and processed data
│   ├── data/                         # AI4I 2020 dataset and preprocessing results
│   ├── models/                       # Optimized neural network implementations
│   └── utils/                        # Common utilities and helper functions
├── Step0-DataAnalysis/               # Phase 0: Exploratory data analysis
│   └── Data_Exploration.ipynb       # Comprehensive dataset analysis notebook
└── Step1-Experiment/                # Phase 1: Model development and federated learning
    ├── central/                      # Centralized baseline model development
    ├── federated/                    # Simulated federated learning experiments
    └── NetworkFed/                   # Real network federated learning implementation
```

## Key Research Contributions

### 1. Novel Hybrid FL-EC Architecture
- First comprehensive integration of federated learning with edge computing for industrial predictive maintenance
- Addresses the gap between theoretical FL research and practical industrial deployment
- Supports both simulation and real network deployment scenarios

### 2. Multi-Modal Neural Network Framework
- **Optimized CNN**: Spatial pattern recognition in multi-sensor industrial data
- **Optimized LSTM**: Temporal dependency modeling for failure prediction
- **Optimized Hybrid (CNN-LSTM)**: Combined spatial-temporal pattern recognition

### 3. Comprehensive Algorithm Evaluation
- **FedAvg**: Baseline federated averaging for collaborative learning
- **FedProx**: Proximal optimization for heterogeneous environments
- **FedDyn**: Dynamic regularization for improved convergence
- **FedNova**: Normalized averaging for system heterogeneity handling

### 4. Statistical Validation Framework
- Support for multiple independent experimental runs (32 runs for statistical significance)
- Comprehensive performance metrics (accuracy, precision, recall, F1-score, AUC-ROC)
- Advanced statistical analysis including ANOVA, post-hoc tests, and effect size calculations

## Dataset and Experimental Design

### AI4I 2020 Predictive Maintenance Dataset
- **Source**: Industrial IoT sensor data from predictive maintenance scenarios
- **Features**: 10 numerical features representing various sensor measurements
- **Target**: Binary classification (failure/no-failure prediction)
- **Samples**: 10,000 data points with realistic industrial failure patterns

### Experimental Configuration
```
Total Experiments = 3 Models × 4 Algorithms × 2 Distributions × 2 Communication Styles = 48 Experiments
```

**Variables**:
- **Models**: OptimizedCNN, OptimizedLSTM, OptimizedHybrid
- **Algorithms**: FedAvg, FedProx, FedDyn, FedNova
- **Data Distributions**: IID (uniform), Non-IID (Dirichlet α=0.5)
- **Communication Styles**: Standard, Secure (with differential privacy)

## Quick Start Guide

### Prerequisites
```bash
# Python 3.8+ with required dependencies
pip install torch torchvision syft pandas numpy scikit-learn matplotlib seaborn scipy statsmodels
```

### Basic Usage

#### 1. Data Analysis (Phase 0)
```bash
cd Step0-DataAnalysis/
jupyter notebook Data_Exploration.ipynb
```

#### 2. Centralized Models (Phase 1a)
```bash
cd Step1-Experiment/central/
jupyter notebook step1a_central_models.ipynb
```

#### 3. Federated Learning Simulation (Phase 1b)
```bash
cd Step1-Experiment/federated/
python step1b_federated_learning_clean_FINAL.py
```

#### 4. Network Federated Learning (Phase 1c)
```bash
cd Step1-Experiment/NetworkFed/
python run_enhanced_experiments.py --run-all --max-rounds 30 --local-epochs 5 --runs 32
```

## Research Questions Addressed

The research systematically addresses five key research questions:

1. **RQ1**: How effectively does Federated Averaging (FedAvg) mitigate privacy risks while ensuring high accuracy in decentralized predictive maintenance models?

2. **RQ2**: What impact does the integration of Federated Learning and Edge Computing have on anomaly detection and feature extraction in sensor data for predictive maintenance in IIoT?

3. **RQ3**: How does the proposed decentralized framework enhance real-time fault detection by modeling temporal dependencies in industrial time-series data while addressing privacy, latency, and scalability challenges?

4. **RQ4**: Does the combination of edge computing and federated learning reduce latency compared to centralized predictive maintenance systems?

5. **RQ5**: How scalable is the proposed framework when tested across diverse industrial setups with varying machine types and sensor configurations?

## Performance Metrics and Evaluation

### Model Performance Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate for failure detection
- **Recall**: Sensitivity to actual failures
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### System Performance Metrics
- **Communication Overhead**: Model update transmission costs
- **Convergence Time**: Rounds required to achieve target performance
- **Latency**: Response time for real-time fault detection
- **Scalability**: Performance degradation with increasing participants

### Privacy Assessment Metrics
- **Differential Privacy Guarantees**: Quantified privacy protection levels
- **Data Leakage Assessment**: Information theoretic privacy analysis
- **Compliance Verification**: GDPR and industrial regulation adherence

## Industrial Applications

### Target Industries
1. **Automotive Manufacturing**: Multi-facility quality control and predictive maintenance coordination
2. **Oil & Gas**: Distributed sensor monitoring across geographically dispersed assets
3. **Manufacturing**: Real-time equipment failure prediction with privacy preservation
4. **Energy**: Smart grid fault detection and preventive maintenance scheduling

### Use Cases
- **Equipment Failure Prediction**: Proactive maintenance scheduling to prevent unexpected downtime
- **Quality Control**: Real-time defect detection in manufacturing processes
- **Asset Health Monitoring**: Continuous surveillance of critical industrial equipment
- **Energy Optimization**: Predictive analytics for energy-efficient operations

## Statistical Analysis and Validation

### Experimental Design
- **Independent Runs**: 32 experimental repetitions for statistical significance
- **Cross-Validation**: K-fold validation within each federated learning scenario
- **Hypothesis Testing**: ANOVA, Friedman test, and post-hoc analyses
- **Effect Size Calculation**: Cohen's d and eta-squared for practical significance

### Statistical Methods
- **Descriptive Statistics**: Mean, standard deviation, confidence intervals
- **Inferential Statistics**: t-tests, ANOVA, non-parametric alternatives
- **Multiple Comparison Correction**: Bonferroni and Tukey HSD adjustments
- **Visualization**: Box plots, violin plots, and statistical significance indicators

## Privacy and Security Framework

### Data Protection Mechanisms
1. **Local Data Retention**: Raw data never leaves individual industrial sites
2. **Differential Privacy**: Calibrated noise injection for enhanced privacy protection
3. **Secure Aggregation**: Cryptographic protection of model updates during transmission
4. **Access Control**: Role-based permissions for federated learning participation

### Compliance and Standards
- **GDPR Compliance**: European data protection regulation adherence
- **Industrial Standards**: IEC 62443 cybersecurity framework alignment
- **Privacy by Design**: Built-in privacy protection mechanisms
- **Audit Trails**: Comprehensive logging for compliance verification

## Future Research Directions

### Technical Enhancements
1. **Advanced Privacy Mechanisms**: Homomorphic encryption and secure multi-party computation
2. **Adaptive Aggregation**: Dynamic weighting based on data quality and reliability
3. **Cross-Platform Deployment**: Support for heterogeneous edge devices and cloud platforms
4. **Real-Time Optimization**: Dynamic resource allocation and load balancing

### Research Extensions
1. **Multi-Modal Integration**: Support for diverse sensor types (vision, audio, environmental)
2. **Transfer Learning**: Cross-domain knowledge transfer between industrial applications
3. **Blockchain Integration**: Immutable audit trails for federated learning transactions
4. **Quantum-Safe Cryptography**: Preparation for post-quantum security requirements

## Citation and Academic Use

If you use this framework in your research, please cite:

```bibtex
@misc{pdm_fdl_framework_2025,
  title={Privacy-Preserving Real-Time Predictive Maintenance in Industrial IoT Systems: A Federated Learning and Edge Computing Approach},
  author={Kiran....},
  year={2025},
  institution={Information Technology Research},
  note={Doctoral Research Framework}
}
```

## Acknowledgments

This research framework implements the methodological approach described in the doctoral research proposal "Integrating Federated Learning and Edge Computing for Privacy-Preserving and Real-time Predictive Maintenance in Industrial IoT Systems."

The work builds upon theoretical foundations from:
- Federated Learning literature (McMahan et al., Li et al.)
- Edge Computing frameworks (Satyanarayanan et al.)
- Industrial IoT predictive maintenance (Lee et al., Zhang et al.)
- Privacy-preserving machine learning (Dwork et al., Kairouz et al.)

---

**Research Phase**: Doctoral Research Implementation  
**Last Updated**: September 2025  
**Framework Version**: 1.0  
**Academic Context**: Doctor of Information Technology Program
