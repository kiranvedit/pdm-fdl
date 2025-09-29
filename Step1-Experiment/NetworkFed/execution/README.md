# Parallel Execution Management

This directory contains the execution management system for coordinating parallel federated learning operations across multiple industrial datasites. The `ParallelExecutionManager` provides robust, fault-tolerant execution with adaptive waiting strategies and real-time health monitoring.

## Architecture Overview

The execution module implements a sophisticated parallel execution framework designed for industrial federated learning environments where datasites may be geographically distributed and subject to network variability, equipment maintenance, and operational constraints.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Parallel Execution Architecture                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                 │
│  │  Heartbeat Manager  │────│   Execution Manager │                 │
│  │  - Health Monitoring│    │   - Parallel Tasks  │                 │
│  │  - Availability     │    │   - Fault Tolerance │                 │
│  │  - Status Tracking  │    │   - Adaptive Waiting │                 │
│  └─────────────────────┘    └─────────────────────┘                 │
│             │                           │                          │
│             ▼                           ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Thread Pool Executor                        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │   │
│  │  │ DataSite A  │ │ DataSite B  │ │ DataSite C  │    ...    │   │
│  │  │ Training    │ │ Training    │ │ Training    │           │   │
│  │  │ Task        │ │ Task        │ │ Task        │           │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│             │               │               │                     │
│             ▼               ▼               ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Results Aggregation & Monitoring              │   │
│  │  - Success/Failure Tracking                                │   │
│  │  - Execution Time Analysis                                 │   │
│  │  - Adaptive Timeout Management                             │   │
│  │  - Statistical Performance Metrics                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Components Overview

### **Parallel Execution Manager (`ParallelExecutionManager`)**

The primary component for managing distributed federated learning execution across multiple industrial datasites with comprehensive fault tolerance and adaptive behavior.

**Key Features:**
- **Thread Pool Execution**: Concurrent training across multiple datasites
- **Adaptive Waiting**: Dynamic timeout adjustment based on datasite health
- **Fault Tolerance**: Robust error handling and recovery mechanisms
- **Real-time Monitoring**: Integration with heartbeat manager for health tracking
- **Statistical Analysis**: Comprehensive execution metrics and performance tracking

---

## 2. Parallel Execution Manager (`ParallelExecutionManager`)

### **What is the Parallel Execution Manager?**

`ParallelExecutionManager` coordinates federated learning rounds across multiple industrial datasites simultaneously, providing resilient execution in environments where datasites may become temporarily unavailable due to maintenance, network issues, or operational constraints.

### **Core Architecture Features**

#### **Initialization and Configuration**
```python
from execution.parallel_execution_manager import ParallelExecutionManager

# Create execution manager with custom settings
execution_manager = ParallelExecutionManager(
    heartbeat_manager=custom_heartbeat_manager,  # Optional: custom heartbeat manager
    max_wait_per_round=600,                      # Maximum wait time per round (seconds)
    check_interval=30                            # Health check interval (seconds)
)
```

**Configuration Parameters:**
- **heartbeat_manager**: Integration with monitoring system for real-time health tracking
- **max_wait_per_round**: Maximum time to wait for round completion (default: 600 seconds)
- **check_interval**: Frequency of datasite health checks (default: 30 seconds)

#### **Execution State Management**
```python
# Internal state tracking
execution_manager.current_round          # Current federated learning round
execution_manager.active_datasites       # List of currently active datasites
execution_manager.round_results         # Results from current round
execution_manager.execution_stats       # Comprehensive execution statistics
```

**Statistical Tracking:**
```python
execution_stats = {
    'rounds_completed': 0,        # Successfully completed rounds
    'total_datasites_used': 0,    # Total datasite participations
    'datasite_failures': 0,       # Count of datasite failures
    'adaptive_waits': 0           # Number of adaptive wait adjustments
}
```

### **Parallel Round Execution**

#### **Main Execution Function**
```python
def execute_parallel_round(self, 
                          training_function, 
                          datasite_configs: List[Dict],
                          round_number: int,
                          **training_kwargs) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute federated learning round in parallel across available datasites.
    
    Returns:
        Tuple of (success_status, comprehensive_results)
    """
```

**Execution Flow:**

**1. Datasite Availability Assessment**
```python
# Check which datasites are currently available
datasite_names = [config['site_id'] for config in datasite_configs]
available_datasites = self.heartbeat_manager.get_available_datasites(datasite_names)

# Ensure minimum participants for federated learning
if len(available_datasites) < 2:
    return False, {}  # Insufficient datasites for federated learning
```

**2. Parallel Task Submission**
```python
# Execute training across all available datasites simultaneously
with ThreadPoolExecutor(max_workers=len(active_configs)) as executor:
    future_to_datasite = {}
    
    for config in active_configs:
        future = executor.submit(
            self._safe_training_execution,
            training_function,
            config,
            round_number,
            **training_kwargs
        )
        future_to_datasite[future] = config['site_id']
```

**3. Adaptive Monitoring and Collection**
```python
# Adaptive waiting with real-time health monitoring
while future_to_datasite and (time.time() - start_time) < self.max_wait_per_round:
    try:
        # Wait for completions with timeout
        for future in as_completed(future_to_datasite.keys(), timeout=self.check_interval):
            datasite_name = future_to_datasite[future]
            result = future.result()
            
            if result['success']:
                round_results[datasite_name] = result['data']
                completed_datasites.add(datasite_name)
                last_completion_time = time.time()
            else:
                failed_datasites.add(datasite_name)
                
    except TimeoutError:
        # No completions in check interval - assess datasite health
        remaining_datasites = list(future_to_datasite.values())
        healthy, still_available = self.heartbeat_manager.are_datasites_healthy(
            remaining_datasites, required_ratio=0.5
        )
        
        if not healthy:
            break  # Too many unhealthy datasites, proceed with completed
            
        # Check for extended wait without progress
        time_since_last_completion = time.time() - last_completion_time
        if time_since_last_completion > (self.max_wait_per_round * 0.5):
            if len(completed_datasites) >= 2:
                break  # Sufficient completions, proceed
```

**4. Results Evaluation and Return**
```python
# Evaluate round success criteria
if len(completed_datasites) >= 2:
    return True, {
        'round_number': round_number,
        'results': round_results,
        'completed_datasites': list(completed_datasites),
        'failed_datasites': list(failed_datasites),
        'execution_time': time.time() - start_time
    }
else:
    return False, {}  # Insufficient completions for aggregation
```

### **Safe Training Execution**

#### **Error-Resistant Training Wrapper**
```python
def _safe_training_execution(self, training_function, datasite_config: Dict, 
                            round_number: int, **kwargs) -> Dict[str, Any]:
    """
    Safely execute training function with comprehensive error handling.
    
    Returns:
        Dict with success status, result data, and metadata
    """
    
    try:
        # Execute training function with provided parameters
        result = training_function(datasite_config, round_number, **kwargs)
        
        return {
            'success': True,
            'data': result,
            'datasite': datasite_config['site_id'],
            'round': round_number,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'datasite': datasite_config['site_id'],
            'round': round_number,
            'timestamp': datetime.now().isoformat()
        }
```

**Error Handling Features:**
- **Exception Isolation**: Individual datasite failures don't affect other datasites
- **Detailed Error Reporting**: Comprehensive error information with timestamps
- **Graceful Degradation**: Continues operation with remaining healthy datasites
- **Metadata Preservation**: Maintains execution context for debugging and analysis

### **Datasite Readiness Management**

#### **Pre-Experiment Validation**
```python
def wait_for_datasites_ready(self, required_datasites: List[str], 
                            minimum_count: int = 2, max_wait: int = 300) -> bool:
    """
    Wait for minimum number of datasites to be ready for experiment.
    
    Args:
        required_datasites: List of required datasite identifiers
        minimum_count: Minimum datasites needed for federated learning
        max_wait: Maximum wait time before timeout
        
    Returns:
        True if sufficient datasites ready, False on timeout
    """
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        available = self.heartbeat_manager.get_available_datasites(required_datasites)
        
        if len(available) >= minimum_count:
            return True
            
        time.sleep(10)  # Check every 10 seconds
    
    return False  # Timeout reached
```

**Readiness Features:**
- **Configurable Minimum**: Flexible minimum datasite requirements
- **Timeout Protection**: Prevents indefinite waiting
- **Progressive Monitoring**: Regular availability checks with logging
- **Early Success Detection**: Returns immediately when requirements met

---

## 3. Usage Patterns and Integration

### **Basic Parallel Execution**

#### **Simple Federated Learning Round**
```python
from execution.parallel_execution_manager import ParallelExecutionManager
from monitoring.heartbeat_manager import get_heartbeat_manager

# Initialize execution manager
execution_manager = ParallelExecutionManager(
    heartbeat_manager=get_heartbeat_manager(),
    max_wait_per_round=900,  # 15 minutes maximum per round
    check_interval=45        # Check health every 45 seconds
)

# Define training function for individual datasites
def federated_training_function(datasite_config, round_number, global_model, **kwargs):
    """Training function executed on each datasite."""
    site_id = datasite_config['site_id']
    
    # Initialize or connect to datasite
    datasite = create_datasite_from_config(datasite_config)
    
    # Perform local training
    training_result = datasite.train_local_model(
        global_model=global_model,
        training_config={
            'round_num': round_number,
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'epochs': kwargs.get('local_epochs', 3),
            'batch_size': kwargs.get('batch_size', 32)
        }
    )
    
    return training_result

# Configure industrial datasites
industrial_datasites = [
    {
        'site_id': 'automotive_detroit',
        'site_name': 'Detroit Automotive Plant',
        'hostname': '192.168.10.10',
        'port': 8080,
        'model_type': 'cnn'
    },
    {
        'site_id': 'chemical_houston',
        'site_name': 'Houston Chemical Facility', 
        'hostname': '192.168.20.15',
        'port': 8081,
        'model_type': 'lstm'
    },
    {
        'site_id': 'aerospace_seattle',
        'site_name': 'Seattle Aerospace Manufacturing',
        'hostname': '192.168.30.20',
        'port': 8082,
        'model_type': 'hybrid'
    }
]

# Execute federated learning round
success, results = execution_manager.execute_parallel_round(
    training_function=federated_training_function,
    datasite_configs=industrial_datasites,
    round_number=1,
    global_model=initial_global_model,
    learning_rate=0.001,
    local_epochs=3,
    batch_size=32
)

if success:
    print(f"✅ Round 1 completed successfully!")
    print(f"Completed datasites: {results['completed_datasites']}")
    print(f"Failed datasites: {results['failed_datasites']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    
    # Process results for model aggregation
    training_results = results['results']
    updated_global_model = aggregate_model_updates(training_results)
else:
    print("❌ Round 1 failed - insufficient datasite participation")
```

### **Multi-Round Federated Learning Campaign**

#### **Complete Federated Learning Experiment**
```python
class IndustrialFederatedLearningCampaign:
    def __init__(self, datasites_config: List[Dict], experiment_config: Dict):
        self.datasites_config = datasites_config
        self.experiment_config = experiment_config
        self.execution_manager = ParallelExecutionManager(
            max_wait_per_round=experiment_config.get('max_wait_per_round', 600),
            check_interval=experiment_config.get('check_interval', 30)
        )
        self.campaign_results = {
            'rounds': [],
            'global_models': [],
            'execution_stats': []
        }
    
    def execute_campaign(self) -> Dict[str, Any]:
        """Execute complete federated learning campaign."""
        
        # Step 1: Wait for minimum datasites to be ready
        datasite_ids = [config['site_id'] for config in self.datasites_config]
        min_datasites = self.experiment_config.get('min_datasites', 2)
        
        self.logger.info(f"Waiting for minimum {min_datasites} datasites...")
        if not self.execution_manager.wait_for_datasites_ready(
            required_datasites=datasite_ids,
            minimum_count=min_datasites,
            max_wait=self.experiment_config.get('startup_wait', 300)
        ):
            raise RuntimeError("Insufficient datasites available for campaign")
        
        # Step 2: Initialize global model
        global_model = self._initialize_global_model()
        
        # Step 3: Execute federated learning rounds
        max_rounds = self.experiment_config.get('max_rounds', 10)
        successful_rounds = 0
        
        for round_num in range(1, max_rounds + 1):
            self.logger.info(f"\n🚀 Starting Round {round_num}/{max_rounds}")
            
            # Execute parallel round
            success, round_results = self.execution_manager.execute_parallel_round(
                training_function=self._federated_training_function,
                datasite_configs=self.datasites_config,
                round_number=round_num,
                global_model=global_model,
                **self.experiment_config.get('training_params', {})
            )
            
            if success:
                successful_rounds += 1
                
                # Store round results
                self.campaign_results['rounds'].append(round_results)
                
                # Aggregate model updates
                training_results = round_results['results']
                global_model = self._aggregate_model_updates(
                    training_results, 
                    self.experiment_config.get('aggregation_algorithm', 'fedavg')
                )
                self.campaign_results['global_models'].append(global_model)
                
                # Log round statistics
                execution_stats = self.execution_manager.get_execution_stats()
                self.campaign_results['execution_stats'].append(execution_stats)
                
                self.logger.info(f"✅ Round {round_num} completed successfully")
                self.logger.info(f"   Completed: {len(round_results['completed_datasites'])}")
                self.logger.info(f"   Failed: {len(round_results['failed_datasites'])}")
                self.logger.info(f"   Execution time: {round_results['execution_time']:.2f}s")
                
                # Check early stopping criteria
                if self._should_stop_early(round_num):
                    self.logger.info(f"Early stopping triggered after round {round_num}")
                    break
                    
            else:
                self.logger.error(f"❌ Round {round_num} failed")
                
                # Check failure tolerance
                failure_rate = 1 - (successful_rounds / round_num)
                max_failure_rate = self.experiment_config.get('max_failure_rate', 0.3)
                
                if failure_rate > max_failure_rate:
                    self.logger.error(f"Campaign failure rate {failure_rate:.2f} exceeds threshold {max_failure_rate}")
                    break
        
        # Step 4: Final campaign evaluation
        campaign_summary = self._generate_campaign_summary(successful_rounds, max_rounds)
        
        return {
            'success': successful_rounds > 0,
            'successful_rounds': successful_rounds,
            'total_rounds': max_rounds,
            'final_global_model': global_model,
            'campaign_results': self.campaign_results,
            'campaign_summary': campaign_summary
        }
    
    def _federated_training_function(self, datasite_config, round_number, global_model, **kwargs):
        """Standardized training function for campaign execution."""
        try:
            # Create or retrieve datasite
            datasite = self._get_or_create_datasite(datasite_config)
            
            # Prepare training configuration
            training_config = {
                'round_num': round_number,
                'model_type': datasite_config.get('model_type', 'cnn'),
                'learning_rate': kwargs.get('learning_rate', 0.001),
                'batch_size': kwargs.get('batch_size', 32),
                'epochs': kwargs.get('local_epochs', 1),
                'algorithm': kwargs.get('algorithm', 'fedavg')
            }
            
            # Execute local training
            training_result = datasite.train_local_model(global_model, training_config)
            
            # Enhanced result with datasite metadata
            enhanced_result = {
                **training_result,
                'datasite_metadata': {
                    'site_id': datasite_config['site_id'],
                    'site_name': datasite_config.get('site_name', 'Unknown'),
                    'model_type': datasite_config.get('model_type', 'cnn'),
                    'round_number': round_number
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Training failed for {datasite_config['site_id']}: {e}")
            raise
```

### **Advanced Execution Strategies**

#### **Adaptive Execution with Dynamic Thresholds**
```python
class AdaptiveExecutionManager:
    def __init__(self, base_execution_manager: ParallelExecutionManager):
        self.execution_manager = base_execution_manager
        self.performance_history = []
        self.dynamic_thresholds = {
            'min_completion_rate': 0.6,  # Minimum successful completion rate
            'adaptive_timeout_factor': 1.0,  # Timeout scaling factor
            'health_check_frequency': 30  # Seconds between health checks
        }
    
    def execute_adaptive_round(self, training_function, datasite_configs, round_number, **kwargs):
        """Execute round with adaptive parameters based on historical performance."""
        
        # Adjust execution parameters based on history
        self._adjust_execution_parameters(round_number)
        
        # Execute round with adapted parameters
        success, results = self.execution_manager.execute_parallel_round(
            training_function, datasite_configs, round_number, **kwargs
        )
        
        # Record performance for future adaptation
        self._record_performance(round_number, success, results)
        
        return success, results
    
    def _adjust_execution_parameters(self, round_number: int):
        """Dynamically adjust execution parameters based on performance history."""
        
        if len(self.performance_history) < 3:
            return  # Need minimum history for adaptation
        
        # Calculate recent performance metrics
        recent_rounds = self.performance_history[-3:]
        avg_completion_rate = sum(r['completion_rate'] for r in recent_rounds) / len(recent_rounds)
        avg_execution_time = sum(r['execution_time'] for r in recent_rounds) / len(recent_rounds)
        
        # Adjust timeout based on recent execution times
        if avg_execution_time > self.execution_manager.max_wait_per_round * 0.8:
            # Increase timeout if rounds are taking longer
            self.dynamic_thresholds['adaptive_timeout_factor'] *= 1.2
            self.execution_manager.max_wait_per_round = int(
                self.execution_manager.max_wait_per_round * self.dynamic_thresholds['adaptive_timeout_factor']
            )
            self.logger.info(f"Round {round_number}: Increased timeout to {self.execution_manager.max_wait_per_round}s")
        
        # Adjust health check frequency based on completion rate
        if avg_completion_rate < self.dynamic_thresholds['min_completion_rate']:
            # More frequent health checks if completion rate is low
            self.execution_manager.check_interval = max(15, self.execution_manager.check_interval - 5)
            self.logger.info(f"Round {round_number}: Increased health check frequency to {self.execution_manager.check_interval}s")
        elif avg_completion_rate > 0.9:
            # Less frequent health checks if performance is excellent
            self.execution_manager.check_interval = min(60, self.execution_manager.check_interval + 5)
            self.logger.info(f"Round {round_number}: Decreased health check frequency to {self.execution_manager.check_interval}s")
    
    def _record_performance(self, round_number: int, success: bool, results: Dict):
        """Record round performance for future adaptive decisions."""
        
        if success and results:
            completion_rate = len(results['completed_datasites']) / (
                len(results['completed_datasites']) + len(results['failed_datasites'])
            )
            execution_time = results['execution_time']
        else:
            completion_rate = 0.0
            execution_time = self.execution_manager.max_wait_per_round
        
        performance_record = {
            'round_number': round_number,
            'success': success,
            'completion_rate': completion_rate,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history (last 10 rounds)
        self.performance_history = self.performance_history[-10:]
```

### **Production Monitoring and Analysis**

#### **Execution Performance Monitor**
```python
class ExecutionPerformanceMonitor:
    def __init__(self, execution_manager: ParallelExecutionManager):
        self.execution_manager = execution_manager
        self.round_metrics = []
        self.datasite_performance = {}
    
    def analyze_execution_performance(self) -> Dict[str, Any]:
        """Comprehensive analysis of execution performance."""
        
        stats = self.execution_manager.get_execution_stats()
        
        # Calculate performance metrics
        if stats['rounds_completed'] > 0:
            avg_datasites_per_round = stats['total_datasites_used'] / stats['rounds_completed']
            failure_rate = stats['datasite_failures'] / stats['total_datasites_used'] if stats['total_datasites_used'] > 0 else 0
            adaptive_wait_rate = stats['adaptive_waits'] / stats['rounds_completed']
        else:
            avg_datasites_per_round = 0
            failure_rate = 0
            adaptive_wait_rate = 0
        
        # Analyze datasite reliability
        datasite_reliability = self._analyze_datasite_reliability()
        
        # Generate performance recommendations
        recommendations = self._generate_performance_recommendations(stats, failure_rate, adaptive_wait_rate)
        
        return {
            'execution_statistics': stats,
            'performance_metrics': {
                'avg_datasites_per_round': avg_datasites_per_round,
                'failure_rate': failure_rate,
                'adaptive_wait_rate': adaptive_wait_rate
            },
            'datasite_reliability': datasite_reliability,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_datasite_reliability(self) -> Dict[str, Any]:
        """Analyze individual datasite performance and reliability."""
        reliability_analysis = {}
        
        for datasite_id, performance_data in self.datasite_performance.items():
            total_attempts = performance_data['successes'] + performance_data['failures']
            success_rate = performance_data['successes'] / total_attempts if total_attempts > 0 else 0
            avg_response_time = sum(performance_data['response_times']) / len(performance_data['response_times']) if performance_data['response_times'] else 0
            
            reliability_analysis[datasite_id] = {
                'success_rate': success_rate,
                'total_attempts': total_attempts,
                'avg_response_time': avg_response_time,
                'reliability_score': success_rate * (1 - min(avg_response_time / 300, 1))  # Factor in response time
            }
        
        return reliability_analysis
    
    def _generate_performance_recommendations(self, stats, failure_rate, adaptive_wait_rate) -> List[str]:
        """Generate actionable performance recommendations."""
        recommendations = []
        
        if failure_rate > 0.2:
            recommendations.append(f"High failure rate ({failure_rate:.2%}) detected. Consider investigating datasite connectivity and health.")
        
        if adaptive_wait_rate > 0.5:
            recommendations.append(f"Frequent adaptive waits ({adaptive_wait_rate:.2%}) suggest network or performance issues. Consider increasing check intervals.")
        
        if stats['rounds_completed'] > 0:
            avg_datasites = stats['total_datasites_used'] / stats['rounds_completed']
            if avg_datasites < 3:
                recommendations.append(f"Low average datasite participation ({avg_datasites:.1f}). Consider adding more datasites or improving availability.")
        
        if not recommendations:
            recommendations.append("Execution performance is optimal. No immediate action required.")
        
        return recommendations
```

---

## 4. Integration with Monitoring Systems

### **Heartbeat Manager Integration**

#### **Health-Aware Execution**
```python
# Integration with monitoring system for real-time health awareness
from monitoring.heartbeat_manager import get_heartbeat_manager

execution_manager = ParallelExecutionManager(
    heartbeat_manager=get_heartbeat_manager(),
    max_wait_per_round=600,
    check_interval=30
)

# Health-aware execution automatically:
# 1. Checks datasite availability before round execution
# 2. Monitors datasite health during execution
# 3. Adapts waiting strategy based on health status
# 4. Provides early termination if too many datasites become unhealthy
```

#### **Custom Health Check Integration**
```python
class CustomHealthAwareExecution:
    def __init__(self, custom_health_checker):
        self.health_checker = custom_health_checker
        self.execution_manager = ParallelExecutionManager(
            heartbeat_manager=custom_health_checker,
            max_wait_per_round=900,
            check_interval=45
        )
    
    def execute_with_custom_health_logic(self, training_function, datasite_configs, round_number, **kwargs):
        """Execute round with custom health checking logic."""
        
        # Pre-execution custom health check
        healthy_sites = self._custom_pre_execution_check(datasite_configs)
        
        if len(healthy_sites) < 2:
            return False, {'error': 'Insufficient healthy datasites'}
        
        # Filter configs to only healthy sites
        healthy_configs = [config for config in datasite_configs if config['site_id'] in healthy_sites]
        
        # Execute with health-filtered datasites
        success, results = self.execution_manager.execute_parallel_round(
            training_function, healthy_configs, round_number, **kwargs
        )
        
        # Post-execution health assessment
        if success:
            self._custom_post_execution_assessment(results)
        
        return success, results
    
    def _custom_pre_execution_check(self, datasite_configs) -> List[str]:
        """Custom logic for pre-execution health validation."""
        healthy_sites = []
        
        for config in datasite_configs:
            site_id = config['site_id']
            
            # Custom health checks (network latency, resource availability, etc.)
            if self._check_network_latency(config) and self._check_resource_availability(config):
                healthy_sites.append(site_id)
        
        return healthy_sites
```

---

## 5. Performance Optimization & Best Practices

### **Optimal Execution Configuration**

#### **Environment-Specific Optimization**
```python
def optimize_execution_for_environment(network_quality: str, datasite_count: int) -> Dict[str, int]:
    """Optimize execution parameters based on deployment environment."""
    
    if network_quality == 'high_speed_lan':
        # Local area network with high-speed connections
        return {
            'max_wait_per_round': 300,    # 5 minutes - fast network
            'check_interval': 15,         # Frequent checks possible
            'thread_pool_size': min(datasite_count, 20)  # Higher concurrency
        }
    
    elif network_quality == 'wan_distributed':
        # Wide area network with geographic distribution
        return {
            'max_wait_per_round': 900,    # 15 minutes - allow for latency
            'check_interval': 45,         # Less frequent checks
            'thread_pool_size': min(datasite_count, 10)  # Moderate concurrency
        }
    
    elif network_quality == 'cellular_edge':
        # Edge computing with cellular connections
        return {
            'max_wait_per_round': 1800,   # 30 minutes - unreliable connections
            'check_interval': 60,         # Infrequent checks to reduce overhead
            'thread_pool_size': min(datasite_count, 5)   # Lower concurrency
        }
    
    else:  # Default conservative settings
        return {
            'max_wait_per_round': 600,    # 10 minutes
            'check_interval': 30,         # Standard checks
            'thread_pool_size': min(datasite_count, 8)   # Moderate concurrency
        }
```

#### **Memory and Resource Management**
```python
class ResourceOptimizedExecution:
    def __init__(self, available_memory_gb: float = 4.0, cpu_cores: int = 4):
        self.available_memory = available_memory_gb
        self.cpu_cores = cpu_cores
        
        # Calculate optimal thread pool size based on resources
        optimal_threads = min(cpu_cores * 2, 16)  # 2x CPU cores, max 16
        
        self.execution_manager = ParallelExecutionManager(
            max_wait_per_round=self._calculate_optimal_timeout(),
            check_interval=30
        )
        
        # Override thread pool size in ThreadPoolExecutor
        self.max_workers = optimal_threads
    
    def _calculate_optimal_timeout(self) -> int:
        """Calculate optimal timeout based on available resources."""
        # More memory = faster processing = shorter timeout needed
        base_timeout = 600  # 10 minutes base
        memory_factor = max(0.5, min(2.0, self.available_memory / 4.0))
        cpu_factor = max(0.5, min(2.0, self.cpu_cores / 4.0))
        
        optimal_timeout = int(base_timeout / (memory_factor * cpu_factor))
        return max(300, optimal_timeout)  # Minimum 5 minutes
    
    def execute_resource_aware_round(self, training_function, datasite_configs, round_number, **kwargs):
        """Execute round with resource-aware optimization."""
        
        # Limit concurrent datasites based on available resources
        memory_per_datasite = 0.5  # GB estimated per datasite
        max_concurrent = int(self.available_memory / memory_per_datasite)
        
        if len(datasite_configs) > max_concurrent:
            # Execute in batches if too many datasites for available memory
            return self._execute_in_batches(training_function, datasite_configs, round_number, max_concurrent, **kwargs)
        else:
            # Execute all at once if memory allows
            return self.execution_manager.execute_parallel_round(training_function, datasite_configs, round_number, **kwargs)
```

---

## 6. Error Handling and Resilience

### **Comprehensive Error Recovery**

#### **Multi-Level Error Handling**
```python
class ResilientExecutionManager:
    def __init__(self, base_manager: ParallelExecutionManager, max_retries: int = 3):
        self.base_manager = base_manager
        self.max_retries = max_retries
        self.error_history = []
    
    def execute_resilient_round(self, training_function, datasite_configs, round_number, **kwargs):
        """Execute round with comprehensive error recovery."""
        
        for attempt in range(self.max_retries + 1):
            try:
                # Attempt execution
                success, results = self.base_manager.execute_parallel_round(
                    training_function, datasite_configs, round_number, **kwargs
                )
                
                if success:
                    # Record successful execution
                    self._record_success(round_number, attempt + 1)
                    return success, results
                else:
                    # Handle execution failure
                    if attempt < self.max_retries:
                        self._handle_execution_failure(round_number, attempt + 1, results)
                        # Modify datasite configs for retry (remove failed datasites)
                        datasite_configs = self._filter_failed_datasites(datasite_configs, results)
                        continue
                    else:
                        self._record_final_failure(round_number, results)
                        return False, results
                        
            except Exception as e:
                self._record_exception(round_number, attempt + 1, e)
                if attempt >= self.max_retries:
                    raise
                # Wait before retry
                time.sleep(30 * (attempt + 1))  # Exponential backoff
        
        return False, {}
    
    def _handle_execution_failure(self, round_number: int, attempt: int, results: Dict):
        """Handle execution failure with targeted recovery."""
        self.logger.warning(f"Round {round_number} attempt {attempt} failed, analyzing failure...")
        
        # Analyze failure patterns
        if results and 'failed_datasites' in results:
            failed_datasites = results['failed_datasites']
            self.logger.info(f"Failed datasites: {failed_datasites}")
            
            # Attempt to identify common failure patterns
            failure_analysis = self._analyze_failure_patterns(failed_datasites)
            self.logger.info(f"Failure analysis: {failure_analysis}")
        
        # Implement recovery strategies
        self._implement_recovery_strategies(round_number, attempt)
    
    def _filter_failed_datasites(self, datasite_configs: List[Dict], results: Dict) -> List[Dict]:
        """Remove consistently failing datasites from subsequent attempts."""
        if not results or 'failed_datasites' not in results:
            return datasite_configs
        
        failed_sites = set(results['failed_datasites'])
        
        # Filter out failed datasites
        filtered_configs = [
            config for config in datasite_configs 
            if config['site_id'] not in failed_sites
        ]
        
        self.logger.info(f"Filtered out {len(failed_sites)} failed datasites for retry")
        return filtered_configs
```

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Execution Engine Version**: 2.0  
**Integration**: Heartbeat Manager, Monitoring Systems  
**Industrial Focus**: Fault-Tolerant Distributed Training Coordination
