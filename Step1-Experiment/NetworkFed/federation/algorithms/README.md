# Federated Learning Algorithms

This directory contains implementations of four state-of-the-art federated learning algorithms designed for industrial IoT predictive maintenance. Each algorithm addresses specific challenges in federated learning environments such as data heterogeneity, system heterogeneity, and convergence stability.

## Algorithm Overview

| Algorithm | Primary Focus | Key Innovation | Use Case |
|-----------|---------------|----------------|-----------|
| **FedAvg** | Baseline federated learning | Weighted averaging | Homogeneous environments |
| **FedProx** | System heterogeneity | Proximal regularization | Heterogeneous client capabilities |
| **FedDyn** | Client drift mitigation | Dynamic regularization | Non-IID data distributions |
| **FedNova** | Objective inconsistency | Normalized averaging | Varying local training intensity |

---

## 1. FedAvg (Federated Averaging)

### What is FedAvg?
FedAvg is the foundational federated learning algorithm proposed by McMahan et al. (2017). It performs weighted averaging of client model updates based on local dataset sizes, providing a simple yet effective approach to collaborative learning without data sharing.

### How FedAvg Works
1. **Client Selection**: Server selects a subset of clients for each round
2. **Local Training**: Each client trains the global model on their local data for E epochs
3. **Upload Updates**: Clients send model parameters to the server
4. **Weighted Aggregation**: Server averages client models weighted by dataset size
5. **Global Update**: Updated global model is distributed to clients

### Mathematical Formulation
```
w_{t+1} = Σ(n_k/n) * w_k^{t+1}
```
Where:
- `w_{t+1}`: Global model at round t+1
- `n_k`: Number of samples at client k
- `n`: Total samples across all clients
- `w_k^{t+1}`: Local model update from client k

### Implementation Parameters
```python
class FedAvgAlgorithm:
    def __init__(self, learning_rate=0.01, device='cpu'):
        self.learning_rate = 0.01    # Global learning rate
        self.local_epochs = 1        # Local training epochs per round
        self.use_proximal_term = False  # No regularization
```

### Why These Parameters?
- **Learning Rate (0.01)**: Balanced convergence speed and stability for industrial sensor data
- **Local Epochs (1)**: Single epoch prevents overfitting on small client datasets
- **No Proximal Term**: Maintains algorithmic simplicity for baseline comparisons

### Key Features
- **Convergence Indicator**: Measures client loss variance to assess convergence
- **Weight Variance Tracking**: Monitors data distribution across clients
- **Participation Statistics**: Tracks client engagement over training rounds

### When to Use FedAvg
- **Homogeneous Environments**: Similar client capabilities and data distributions
- **Baseline Experiments**: Reference point for comparing advanced algorithms
- **Simple Deployments**: Minimal computational overhead required

---

## 2. FedProx (Federated Proximal)

### What is FedProx?
FedProx extends FedAvg by adding a proximal term during local training to handle system heterogeneity. It addresses challenges when clients have varying computational capabilities and unreliable participation patterns.

### How FedProx Works
1. **Proximal Regularization**: Adds penalty term to keep local updates close to global model
2. **Adaptive Local Training**: Clients can perform variable amounts of local training
3. **Heterogeneity Handling**: Tolerates partial client participation and dropouts
4. **Weighted Aggregation**: Same as FedAvg but with regularized local updates

### Mathematical Formulation
```
Local Loss: L_k(w) + (μ/2) * ||w - w^t||^2
Global Update: w_{t+1} = Σ(n_k/n) * w_k^{t+1}
```
Where:
- `μ`: Proximal term coefficient (regularization strength)
- `w^t`: Global model at round t (proximal center)
- `L_k(w)`: Local loss function at client k

### Implementation Parameters
```python
class FedProxAlgorithm:
    def __init__(self, learning_rate=0.01, mu=0.01, device='cpu'):
        self.learning_rate = 0.01    # Global learning rate
        self.mu = 0.01              # Proximal term coefficient
        self.local_epochs = 1       # Local training epochs
        self.use_proximal_term = True  # Enable proximal regularization
```

### Why These Parameters?
- **Proximal Coefficient μ (0.01)**: 
  - Balances local adaptation vs. global consistency
  - Small value allows client customization while preventing drift
  - Empirically optimal for industrial IoT scenarios with moderate heterogeneity
- **Learning Rate (0.01)**: Consistent with FedAvg for fair comparison
- **Single Local Epoch**: Reduces computational burden on resource-constrained industrial devices

### Key Features
- **Client Drift Tracking**: Monitors how much each client deviates from global model
- **Proximal Loss Monitoring**: Measures regularization effect on training
- **Regularization Effect Calculation**: Quantifies impact of proximal term on convergence
- **Adaptive μ Updates**: Supports dynamic adjustment of regularization strength

### When to Use FedProx
- **Heterogeneous Systems**: Clients with varying computational capabilities
- **Unreliable Networks**: Intermittent client participation patterns
- **Resource Constraints**: Limited computational resources on edge devices
- **Non-IID Data**: Moderate data heterogeneity across industrial sites

---

## 3. FedDyn (Federated Dynamic Regularization)

### What is FedDyn?
FedDyn introduces dynamic regularization to address client drift and improve convergence in highly heterogeneous federated learning environments. It maintains a global regularization parameter that evolves based on client update patterns.

### How FedDyn Works
1. **Dynamic Regularization**: Maintains evolving global parameter `h` for regularization
2. **Client-Specific Updates**: Tracks individual client regularization states
3. **Gradient Diversity**: Measures client heterogeneity for adaptive regularization
4. **Correction Application**: Applies FedDyn correction to aggregated parameters
5. **Stabilization**: Implements gradient clipping and norm bounds for numerical stability

### Mathematical Formulation
```
Local Training: min L_k(w) + h_k^T * w
Global Update: h^{t+1} = h^t - α * (w^{t+1} - w_k^{t+1})
Correction: w^{t+1} = w_agg - η * h^{t+1}
```
Where:
- `h`: Global dynamic regularization parameter
- `α`: Dynamic regularization coefficient
- `η`: Learning rate for correction
- `w_agg`: Standard aggregated parameters

### Implementation Parameters
```python
class FedDynAlgorithm:
    def __init__(self, learning_rate=0.01, alpha=0.01, device='cpu'):
        self.learning_rate = 0.01     # Global learning rate
        self.alpha = 0.01            # Dynamic regularization coefficient (clamped: 0.001-0.1)
        self.gradient_clipping = 10.0 # Gradient norm threshold
        self.h_clipping = 50.0       # Global h parameter clipping
```

### Why These Parameters?
- **Alpha Coefficient (0.01)**:
  - Controls strength of dynamic regularization
  - Clamped between 0.001-0.1 to prevent numerical instability
  - Optimal for industrial IoT with moderate client drift
- **Gradient Clipping (10.0)**:
  - Prevents gradient explosion in heterogeneous environments
  - Essential for stable training with industrial sensor noise
- **H Parameter Clipping (50.0)**:
  - Bounds global regularization parameter accumulation
  - Prevents indefinite growth that could destabilize training

### Key Features
- **Gradient Diversity Calculation**: Measures client heterogeneity using pairwise update differences
- **Multi-Stage Stabilization**: Gradient clipping → H-delta clipping → Global H clipping
- **Client State Tracking**: Maintains per-client regularization histories
- **Convergence Monitoring**: Tracks regularization stability over rounds

### Advanced Stability Features
- **Numerical Safety**: All arithmetic performed in float32 with overflow protection
- **Adaptive Correction**: Correction magnitude limited to 10% of parameter norm
- **Memory Management**: Maintains only last 10 rounds of metrics for efficiency

### When to Use FedDyn
- **High Data Heterogeneity**: Significant non-IID data across industrial sites
- **Client Drift Issues**: Clients diverging significantly from global objective
- **Complex Industrial Scenarios**: Multi-factory setups with different equipment types
- **Long Training Periods**: Extended federated learning campaigns requiring stability

---

## 4. FedNova (Federated Normalized Averaging)

### What is FedNova?
FedNova addresses objective inconsistency in federated learning by normalizing client updates based on their local training intensity. It handles scenarios where clients perform different amounts of local computation due to varying resources or data availability.

### How FedNova Works
1. **Effective Tau Calculation**: Measures actual local training intensity per client
2. **Update Normalization**: Normalizes client updates based on training effectiveness
3. **Variance-Reduced Averaging**: Applies weighted averaging with variance reduction
4. **Momentum Application**: Incorporates momentum for accelerated convergence
5. **Quality Assessment**: Tracks normalization consistency across rounds

### Mathematical Formulation
```
Effective Tau: τ_k^eff = τ_k * effectiveness_factor * sample_weight
Normalization: Δw_k^norm = (τ_k^eff / τ^global) * Δw_k
Aggregation: w^{t+1} = Σ(weight_k * Δw_k^norm) / Σ(weight_k)
Momentum: v^{t+1} = β * v^t + (1-β) * w^{t+1}
```
Where:
- `τ_k^eff`: Effective local training intensity for client k
- `β`: Momentum coefficient
- `effectiveness_factor`: Training quality measure

### Implementation Parameters
```python
class FedNovaAlgorithm:
    def __init__(self, learning_rate=0.01, momentum=0.9, device='cpu'):
        self.learning_rate = 0.01    # Global learning rate
        self.momentum = 0.9         # Momentum coefficient for acceleration
        self.local_epochs = 1       # Base local epochs
        self.variance_reduction = True  # Enable variance reduction
```

### Why These Parameters?
- **Momentum (0.9)**:
  - Accelerates convergence by maintaining update direction
  - Standard value for federated learning with industrial time-series data
  - Smooths out noise from heterogeneous client updates
- **Learning Rate (0.01)**: Consistent with other algorithms for fair comparison
- **Variance Reduction**: Essential for handling client update inconsistencies
- **Effective Tau Tracking**: Adapts to real training intensity rather than nominal epochs

### Key Features
- **Training Effectiveness Measurement**: Accounts for actual learning progress vs. nominal training
- **Dynamic Normalization**: Adjusts for client computational variations
- **Momentum Buffer**: Maintains velocity for accelerated convergence
- **Variance Reduction Factor**: Quantifies update consistency improvements

### Advanced Normalization Features
- **Loss-Based Effectiveness**: Uses training loss improvement to measure actual training quality
- **Sample-Weighted Tau**: Incorporates dataset size into training intensity calculation
- **Consistency Tracking**: Monitors normalization factor stability over rounds
- **Convergence Stability**: Measures training consistency through variance reduction trends

### When to Use FedNova
- **Variable Client Resources**: Clients with different computational capabilities
- **Inconsistent Training Schedules**: Clients performing different amounts of local training
- **Quality-Focused Scenarios**: Emphasis on training effectiveness over quantity
- **Accelerated Convergence**: When faster convergence is critical for industrial applications

---

## Algorithm Selection Guide

### For Different Industrial Scenarios

#### **Manufacturing Floor (Homogeneous Environment)**
- **Recommended**: FedAvg
- **Rationale**: Similar equipment, consistent data patterns, reliable connectivity
- **Configuration**: Standard parameters with single local epoch

#### **Multi-Site Manufacturing (Moderate Heterogeneity)**
- **Recommended**: FedProx
- **Rationale**: Different sites may have varying capabilities, intermittent connectivity
- **Configuration**: μ=0.01 for balanced local adaptation

#### **Cross-Industry Deployment (High Heterogeneity)**
- **Recommended**: FedDyn
- **Rationale**: Significant data and system differences, client drift concerns
- **Configuration**: α=0.01 with stabilization features enabled

#### **Resource-Constrained Edge Deployment**
- **Recommended**: FedNova
- **Rationale**: Variable computational resources, emphasis on training effectiveness
- **Configuration**: Momentum=0.9 for accelerated convergence

### Performance Comparison

| Metric | FedAvg | FedProx | FedDyn | FedNova |
|--------|--------|---------|--------|---------|
| **Convergence Speed** | Medium | Medium | Slow (Stable) | Fast |
| **Heterogeneity Tolerance** | Low | Medium | High | High |
| **Computational Overhead** | Low | Low | Medium | Medium |
| **Memory Requirements** | Low | Low | High | Medium |
| **Numerical Stability** | High | High | Medium | High |
| **Industrial Applicability** | Basic | Standard | Advanced | Specialized |

### Implementation Notes

#### **Common Features Across All Algorithms**
- **Type Safety**: All algorithms handle mixed tensor dtypes (float32 aggregation, dtype restoration)
- **Error Handling**: Comprehensive exception handling and numerical stability checks
- **Metrics Tracking**: Algorithm-specific metrics for monitoring and debugging
- **State Management**: Full state persistence and reset capabilities

#### **Performance Optimizations**
- **Memory Efficiency**: Limited history tracking (10 rounds maximum)
- **Numerical Stability**: Gradient clipping, parameter bounding, overflow protection
- **Computation Efficiency**: Float32 arithmetic with dtype preservation

#### **Industrial Deployment Considerations**
- **Network Efficiency**: Minimal communication overhead
- **Fault Tolerance**: Graceful handling of client failures and incomplete updates
- **Monitoring Integration**: Rich metrics for production monitoring and debugging
- **Scalability**: Efficient handling of varying numbers of participating clients

---

## Usage Examples

### Basic Algorithm Instantiation
```python
# FedAvg - Baseline federated learning
fedavg = FedAvgAlgorithm(learning_rate=0.01, device='cuda')

# FedProx - Handle system heterogeneity
fedprox = FedProxAlgorithm(learning_rate=0.01, mu=0.01, device='cuda')

# FedDyn - Address client drift
feddyn = FedDynAlgorithm(learning_rate=0.01, alpha=0.01, device='cuda')

# FedNova - Normalize varying training intensity
fednova = FedNovaAlgorithm(learning_rate=0.01, momentum=0.9, device='cuda')
```

### Algorithm Configuration for Industrial IoT
```python
# Industrial manufacturing scenario
config = {
    'fedavg': {'learning_rate': 0.01, 'local_epochs': 1},
    'fedprox': {'learning_rate': 0.01, 'mu': 0.01, 'local_epochs': 1},
    'feddyn': {'learning_rate': 0.01, 'alpha': 0.01, 'gradient_clipping': 10.0},
    'fednova': {'learning_rate': 0.01, 'momentum': 0.9, 'variance_reduction': True}
}
```

### Monitoring Algorithm Performance
```python
# Get algorithm-specific metrics
fedavg_metrics = fedavg.get_algorithm_specific_metrics(client_updates)
fedprox_metrics = fedprox.get_algorithm_specific_metrics(client_updates)
feddyn_metrics = feddyn.get_algorithm_specific_metrics(client_updates)
fednova_metrics = fednova.get_algorithm_specific_metrics(client_updates)

# Track convergence indicators
convergence_indicators = {
    'fedavg': fedavg_metrics.get('fedavg_convergence_indicator', 0.0),
    'fedprox': fedprox_metrics.get('regularization_effect', 0.0),
    'feddyn': feddyn_metrics.get('convergence_indicator', 0.0),
    'fednova': fednova_metrics.get('convergence_stability', 0.0)
}
```

---

## Research References

1. **FedAvg**: McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017.

2. **FedProx**: Li, T., et al. "Federated optimization in heterogeneous networks." MLSys 2020.

3. **FedDyn**: Acar, D. A. E., et al. "Federated learning based on dynamic regularization." ICLR 2021.

4. **FedNova**: Wang, J., et al. "Tackling the objective inconsistency problem in heterogeneous federated optimization." NeurIPS 2020.

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Framework Version**: 2.0  
**Research Phase**: Step 1 - Network Federated Learning Implementation
