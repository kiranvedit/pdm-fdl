# NetworkFed Monitoring System

This directory contains a comprehensive monitoring infrastructure for federated learning experiments, providing real-time tracking of datasite availability, experiment progress, performance metrics, and visual dashboard interfaces. The monitoring system is designed for production-scale federated learning deployments across distributed manufacturing environments.

## Architecture Overview

The monitoring system implements a multi-layered approach to track and visualize federated learning experiments in real-time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NetworkFed Monitoring Architecture              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Web Dashboard Layer                        │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │   │
│  │  │ Experiment      │ │ Real-time       │ │ Interactive   │ │   │
│  │  │ Progress        │ │ Datasite Status │ │ Refresh       │ │   │
│  │  │ - Round Metrics │ │ - Online/Offline│ │ - Auto-update │ │   │
│  │  │ - Performance   │ │ - Heartbeat     │ │ - Manual      │ │   │
│  │  │ - ETA Display   │ │ - Connection    │ │ - Status API  │ │   │
│  │  └─────────────────┘ └─────────────────┘ └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Metrics Collection Layer                    │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │   │
│  │  │ Performance     │ │ Communication   │ │ Training      │ │   │
│  │  │ Metrics         │ │ Metrics         │ │ Metrics       │ │   │
│  │  │ - Accuracy      │ │ - Comm Time     │ │ - Train Time  │ │   │
│  │  │ - Loss Tracking │ │ - Data Transfer │ │ - Local Epochs│ │   │
│  │  │ - Convergence   │ │ - Efficiency    │ │ - Resources   │ │   │
│  │  │ - Model Stats   │ │ - Model Size    │ │ - Throughput  │ │   │
│  │  └─────────────────┘ └─────────────────┘ └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Heartbeat Management Layer                  │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │   │
│  │  │ Availability    │ │ Connection      │ │ Health        │ │   │
│  │  │ Tracking        │ │ Monitoring      │ │ Assessment    │ │   │
│  │  │ - Online Status │ │ - HTTP Server   │ │ - Uptime      │ │   │
│  │  │ - Offline Detect│ │ - REST API      │ │ - Reliability │ │   │
│  │  │ - Timeout Mgmt  │ │ - JSON Protocol │ │ - Min Clients │ │   │
│  │  └─────────────────┘ └─────────────────┘ └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Factory Datasite Network                     │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │   │
│  │  │ Automotive      │ │ Chemical        │ │ Aerospace     │ │   │
│  │  │ Detroit Factory │ │ Houston Plant   │ │ Seattle Mfg   │ │   │
│  │  │ - Heartbeat TX  │ │ - Heartbeat TX  │ │ - Heartbeat TX│ │   │
│  │  │ - Status Report │ │ - Status Report │ │ - Status Rpt  │ │   │
│  │  │ - Health Check  │ │ - Health Check  │ │ - Health Chk  │ │   │
│  │  └─────────────────┘ └─────────────────┘ └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Components Overview

### **Module Initialization (`__init__.py`)**
```python
from .metrics_collector import MetricsCollector

__all__ = [
    'MetricsCollector'
]
```

**Exported Components:**
- **MetricsCollector**: Primary interface for comprehensive metrics collection during federated learning

---

## 2. Comprehensive Metrics Collection (`MetricsCollector`)

### **What is the MetricsCollector?**

The `MetricsCollector` provides comprehensive tracking and analysis of federated learning experiments, collecting performance, communication, training, and convergence metrics across all federated learning rounds with automated analysis and reporting capabilities.

### **Core Metrics Collection Features**

#### **Round-Level Metrics Collection**
```python
from monitoring import MetricsCollector

# Initialize metrics collector
metrics_collector = MetricsCollector()

# Collect comprehensive metrics for each federated round
def execute_monitored_federated_round(round_num, global_model, client_updates):
    """Execute federated round with comprehensive monitoring."""
    
    # Collect round metrics
    round_metrics = metrics_collector.collect_round_metrics(
        round_num=round_num,
        global_model=global_model,
        client_updates=client_updates
    )
    
    # Round metrics structure
    example_round_metrics = {
        'round_num': 7,
        'timestamp': 1725543015.123456,
        'num_participants': 8,
        'total_samples': 15240,
        
        # Performance metrics
        'avg_accuracy': 0.9215,
        'avg_loss': 0.0845,
        'avg_precision': 0.9180,
        'avg_recall': 0.9250,
        'avg_f1_score': 0.9215,
        'std_accuracy': 0.0145,
        'min_accuracy': 0.8980,
        'max_accuracy': 0.9420,
        'client_performance_variance': 0.000210,
        
        # Communication metrics
        'total_communication_time': 45.67,
        'avg_communication_time': 5.71,
        'max_communication_time': 8.45,
        'total_data_transferred_mb': 124.8,
        'avg_model_size_mb': 15.6,
        'communication_efficiency': 0.175,
        
        # Training metrics
        'total_training_time': 234.15,
        'avg_training_time': 29.27,
        'max_training_time': 45.60,
        'avg_local_epochs': 3.2,
        'training_efficiency': 0.109,
        
        # Model metrics
        'model_total_parameters': 452890,
        'model_size_mb': 15.6,
        'model_avg_param_norm': 2.34,
        'model_max_param_norm': 8.92,
        'model_param_variance': 1.23,
        
        # Convergence metrics
        'convergence_rate': 0.0125,
        'is_converging': True,
        'stability_metric': 85.4,
        'accuracy_improvement': True
    }
    
    return round_metrics
```

**Round Metrics Categories:**
- **Performance Metrics**: Accuracy, loss, precision, recall, F1-score with statistical analysis
- **Communication Metrics**: Transfer times, data volumes, efficiency calculations
- **Training Metrics**: Training times, epochs, resource utilization efficiency
- **Model Metrics**: Parameter counts, sizes, norm statistics
- **Convergence Metrics**: Improvement rates, stability analysis, trend detection

### **Performance Analytics**

#### **Weighted Performance Aggregation**
```python
def analyze_client_performance_distribution(client_updates):
    """Analyze performance distribution across federated clients."""
    
    # Extract client performance data
    client_performance = {}
    
    for update in client_updates:
        client_id = update.get('client_id', 'unknown')
        client_performance[client_id] = {
            'accuracy': update.get('accuracy', 0.0),
            'loss': update.get('loss', float('inf')),
            'num_samples': update.get('num_samples', 0),
            'training_time': update.get('training_time', 0.0),
            'model_size': update.get('model_size_mb', 0.0)
        }
    
    # Calculate weighted averages by sample count
    total_samples = sum(cp['num_samples'] for cp in client_performance.values())
    
    weighted_metrics = {
        'weighted_accuracy': sum(
            cp['accuracy'] * cp['num_samples'] / total_samples 
            for cp in client_performance.values()
        ),
        'weighted_loss': sum(
            cp['loss'] * cp['num_samples'] / total_samples 
            for cp in client_performance.values()
        ),
        'performance_variance': calculate_performance_variance(client_performance),
        'client_contribution_analysis': analyze_client_contributions(client_performance)
    }
    
    return weighted_metrics

def calculate_performance_variance(client_performance):
    """Calculate variance in client performance."""
    accuracies = [cp['accuracy'] for cp in client_performance.values()]
    return {
        'accuracy_std': np.std(accuracies),
        'accuracy_range': max(accuracies) - min(accuracies),
        'performance_consistency': 1.0 / (np.var(accuracies) + 1e-8)
    }
```

### **Communication Efficiency Analysis**

#### **Bandwidth and Transfer Optimization Tracking**
```python
def track_communication_efficiency(round_metrics_history):
    """Track communication efficiency trends over federated rounds."""
    
    if len(round_metrics_history) < 2:
        return {'status': 'insufficient_data'}
    
    # Extract communication metrics over time
    communication_trends = {
        'rounds': [rm['round_num'] for rm in round_metrics_history],
        'comm_times': [rm['total_communication_time'] for rm in round_metrics_history],
        'data_transfer': [rm['total_data_transferred_mb'] for rm in round_metrics_history],
        'efficiency_scores': [rm['communication_efficiency'] for rm in round_metrics_history]
    }
    
    # Calculate efficiency trends
    recent_efficiency = np.mean(communication_trends['efficiency_scores'][-5:])
    early_efficiency = np.mean(communication_trends['efficiency_scores'][:5])
    efficiency_improvement = recent_efficiency - early_efficiency
    
    # Bandwidth utilization analysis
    avg_round_transfer = np.mean(communication_trends['data_transfer'])
    transfer_variance = np.var(communication_trends['data_transfer'])
    
    # Communication health assessment
    communication_health = {
        'efficiency_trend': 'improving' if efficiency_improvement > 0 else 'declining',
        'efficiency_improvement': efficiency_improvement,
        'avg_data_per_round_mb': avg_round_transfer,
        'transfer_consistency': 1.0 / (transfer_variance + 1e-8),
        'communication_stability': assess_communication_stability(communication_trends),
        'recommendations': generate_communication_recommendations(communication_trends)
    }
    
    return communication_health
```

### **Convergence Analysis**

#### **Advanced Convergence Detection and Prediction**
```python
def analyze_convergence_patterns(metrics_collector):
    """Analyze convergence patterns and predict completion."""
    
    performance_history = metrics_collector.performance_history
    
    if len(performance_history) < 5:
        return {'status': 'insufficient_data_for_analysis'}
    
    # Extract accuracy progression
    rounds = [ph['round'] for ph in performance_history]
    accuracies = [ph['accuracy'] for ph in performance_history]
    losses = [ph['loss'] for ph in performance_history]
    
    # Convergence pattern analysis
    convergence_analysis = {
        'current_accuracy': accuracies[-1],
        'best_accuracy': max(accuracies),
        'accuracy_plateau_detection': detect_accuracy_plateau(accuracies),
        'convergence_velocity': calculate_convergence_velocity(accuracies),
        'loss_stability': analyze_loss_stability(losses),
        'estimated_convergence_round': predict_convergence_round(accuracies, rounds),
        'improvement_potential': assess_improvement_potential(accuracies)
    }
    
    return convergence_analysis

def detect_accuracy_plateau(accuracies, window_size=5, threshold=0.001):
    """Detect if accuracy has plateaued."""
    if len(accuracies) < window_size:
        return False
    
    recent_window = accuracies[-window_size:]
    accuracy_range = max(recent_window) - min(recent_window)
    
    return accuracy_range < threshold

def calculate_convergence_velocity(accuracies, window_size=3):
    """Calculate rate of convergence improvement."""
    if len(accuracies) < window_size + 1:
        return 0.0
    
    recent_improvements = []
    for i in range(len(accuracies) - window_size, len(accuracies)):
        if i > 0:
            improvement = accuracies[i] - accuracies[i-1]
            recent_improvements.append(improvement)
    
    return np.mean(recent_improvements) if recent_improvements else 0.0
```

---

## 3. Heartbeat Management System (`HeartbeatManager`)

### **What is the HeartbeatManager?**

The `HeartbeatManager` provides real-time monitoring of factory datasite availability through HTTP-based heartbeat signals, enabling dynamic participant selection and experiment resilience in distributed manufacturing environments.

### **Real-Time Availability Monitoring**

#### **Heartbeat Reception and Processing**
```python
from monitoring.heartbeat_manager import HeartbeatManager, start_heartbeat_manager

# Initialize and start heartbeat manager
heartbeat_manager = start_heartbeat_manager()

# Factory datasites send heartbeat signals
def factory_heartbeat_protocol():
    """Example heartbeat protocol from factory datasite."""
    
    import requests
    import time
    import json
    
    # Factory datasite configuration
    factory_config = {
        'factory_name': 'automotive_detroit_main',
        'heartbeat_interval': 60,  # seconds
        'orchestrator_url': 'http://orchestrator.pdm-network.com:8888'
    }
    
    while True:
        try:
            # Send heartbeat signal
            heartbeat_data = {
                'factory_name': factory_config['factory_name'],
                'timestamp': time.time(),
                'status': 'operational',
                'available_resources': {
                    'cpu_usage': 0.65,
                    'memory_usage': 0.72,
                    'storage_available_gb': 120.5,
                    'active_models': 3
                },
                'last_training_completion': time.time() - 300  # 5 minutes ago
            }
            
            response = requests.post(
                f"{factory_config['orchestrator_url']}/heartbeat",
                json=heartbeat_data,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Heartbeat sent successfully from {factory_config['factory_name']}")
            else:
                print(f"❌ Heartbeat failed: {response.status_code}")
                
        except Exception as e:
            print(f"🚨 Heartbeat error: {e}")
        
        time.sleep(factory_config['heartbeat_interval'])
```

**Heartbeat Protocol Features:**
- **HTTP REST API**: Simple JSON-based heartbeat reception
- **Automatic Status Tracking**: Online/offline status with timeout detection
- **Background Cleanup**: Automatic detection of offline datasites
- **Thread-Safe Operations**: Concurrent heartbeat processing with locking

#### **Dynamic Datasite Availability Management**
```python
def manage_federated_experiment_with_heartbeats(heartbeat_manager, experiment_config):
    """Manage federated experiment with dynamic datasite availability."""
    
    required_datasites = experiment_config['required_datasites']
    minimum_participants = experiment_config.get('minimum_participants', 3)
    
    # Wait for minimum datasites to become available
    print(f"🕐 Waiting for minimum {minimum_participants} datasites...")
    
    if heartbeat_manager.wait_for_minimum_datasites(
        minimum_count=minimum_participants,
        max_wait_seconds=300  # 5 minutes
    ):
        print("✅ Minimum datasites available, starting experiment")
        
        # Get currently available datasites
        available_datasites = heartbeat_manager.get_available_datasites(required_datasites)
        print(f"📡 Available datasites: {available_datasites}")
        
        # Execute federated learning rounds
        for round_num in range(experiment_config['max_rounds']):
            # Check datasite health before each round
            is_healthy, current_available = heartbeat_manager.are_datasites_healthy(
                datasite_names=available_datasites,
                required_ratio=0.7  # At least 70% must be available
            )
            
            if not is_healthy:
                print(f"⚠️ Insufficient healthy datasites for round {round_num}")
                print(f"Currently available: {current_available}")
                
                # Wait for datasites to recover or abort
                if not heartbeat_manager.wait_for_minimum_datasites(minimum_participants, 120):
                    print("🚨 Aborting experiment due to insufficient datasites")
                    break
                
                # Update available datasites list
                available_datasites = current_available
            
            # Execute federated round with available datasites
            round_result = execute_federated_round(round_num, current_available)
            print(f"✅ Round {round_num} completed with {len(current_available)} datasites")
    
    else:
        print("❌ Timeout waiting for minimum datasites, experiment aborted")
```

### **Health Assessment and Resilience**

#### **Datasite Registration and Manual Management**
```python
def setup_factory_network_monitoring(factory_network_config):
    """Setup monitoring for a factory network."""
    
    heartbeat_manager = HeartbeatManager(port=8888, heartbeat_timeout=90)
    heartbeat_manager.start()
    
    # Register known factory datasites
    for factory_name, factory_config in factory_network_config.items():
        heartbeat_manager.register_datasite(
            factory_name=factory_name,
            endpoint=factory_config.get('endpoint'),
            status='offline'  # Start as offline until heartbeat received
        )
        
        print(f"📝 Registered factory datasite: {factory_name}")
    
    return heartbeat_manager

# Example factory network configuration
factory_network = {
    'automotive_detroit_main': {
        'endpoint': 'http://detroit-factory.automotive.com:8080',
        'location': 'Detroit, MI',
        'specialty': 'Engine Manufacturing',
        'expected_uptime': '24/7'
    },
    'chemical_houston_reactor': {
        'endpoint': 'http://houston-plant.chemical.com:8080',
        'location': 'Houston, TX',
        'specialty': 'Process Monitoring',
        'expected_uptime': 'Business Hours'
    },
    'aerospace_seattle_assembly': {
        'endpoint': 'http://seattle-factory.aerospace.com:8080',
        'location': 'Seattle, WA',
        'specialty': 'Component Assembly',
        'expected_uptime': '24/7'
    }
}

# Setup monitoring
hb_manager = setup_factory_network_monitoring(factory_network)
```

#### **Datasite Status Monitoring and Analysis**
```python
def monitor_datasite_reliability(heartbeat_manager, monitoring_duration_hours=24):
    """Monitor datasite reliability over time."""
    
    monitoring_start = time.time()
    reliability_data = {}
    
    while time.time() - monitoring_start < monitoring_duration_hours * 3600:
        # Get current status
        all_status = heartbeat_manager.get_all_datasite_status()
        
        # Update reliability tracking
        current_time = time.time()
        for datasite_id, status in all_status.items():
            if datasite_id not in reliability_data:
                reliability_data[datasite_id] = {
                    'total_checks': 0,
                    'online_checks': 0,
                    'uptime_percentage': 0.0,
                    'last_seen': None,
                    'downtime_events': []
                }
            
            reliability_data[datasite_id]['total_checks'] += 1
            
            if status['status'] == 'online':
                reliability_data[datasite_id]['online_checks'] += 1
                reliability_data[datasite_id]['last_seen'] = current_time
            else:
                # Track downtime event
                if reliability_data[datasite_id]['last_seen']:
                    downtime_duration = current_time - reliability_data[datasite_id]['last_seen']
                    reliability_data[datasite_id]['downtime_events'].append({
                        'start_time': reliability_data[datasite_id]['last_seen'],
                        'duration_minutes': downtime_duration / 60
                    })
            
            # Calculate uptime percentage
            reliability_data[datasite_id]['uptime_percentage'] = (
                reliability_data[datasite_id]['online_checks'] / 
                reliability_data[datasite_id]['total_checks'] * 100
            )
        
        time.sleep(60)  # Check every minute
    
    return reliability_data
```

---

## 4. Interactive Status Dashboard (`SimpleStatusDashboard`)

### **What is the SimpleStatusDashboard?**

The `SimpleStatusDashboard` provides a real-time web-based interface for monitoring federated learning experiments, displaying datasite status, experiment progress, performance metrics, and system health in an intuitive visual format.

### **Real-Time Web Dashboard**

#### **Dashboard Initialization and Setup**
```python
from monitoring.status_dashboard import start_status_dashboard

# Start dashboard on specified port
dashboard = start_status_dashboard(port=8889)

# Dashboard accessible at: http://localhost:8889
print("📊 Status dashboard available at: http://localhost:8889")

# Update experiment status during federated learning
def run_federated_experiment_with_dashboard(experiment_config):
    """Run federated experiment with dashboard monitoring."""
    
    # Update dashboard with experiment start
    dashboard.update_experiment_status(
        current_experiment=experiment_config['experiment_name'],
        current_round=0,
        total_rounds=experiment_config['max_rounds'],
        start_time=datetime.now().isoformat(),
        status='starting'
    )
    
    # Execute federated rounds
    for round_num in range(1, experiment_config['max_rounds'] + 1):
        # Update dashboard with round progress
        dashboard.update_experiment_status(
            current_round=round_num,
            status='training'
        )
        
        # Execute federated round
        round_result = execute_federated_round(round_num)
        
        # Update dashboard with round completion
        dashboard.update_experiment_status(
            completed_experiments=dashboard.experiment_status.get('completed_experiments', 0) + 1
        )
    
    # Update dashboard with experiment completion
    dashboard.update_experiment_status(
        status='completed',
        end_time=datetime.now().isoformat()
    )
```

**Dashboard Features:**
- **Real-Time Updates**: Auto-refresh every 10 seconds with manual refresh capability
- **Experiment Progress**: Round-by-round progress tracking with ETA estimation
- **Datasite Monitoring**: Live online/offline status with heartbeat tracking
- **Performance Metrics**: Success rates, training efficiency, system health
- **Visual Indicators**: Color-coded status, progress bars, animated indicators

#### **Dashboard API and Data Structure**
```python
# Dashboard API endpoint response structure
dashboard_api_response = {
    'datasites': {
        'automotive_detroit_main': {
            'status': 'online',
            'last_heartbeat': '2025-09-05T14:30:15.123456',
            'heartbeat_count': 47,
            'type': 'Factory'
        },
        'chemical_houston_reactor': {
            'status': 'offline',
            'last_heartbeat': '2025-09-05T14:28:42.789012',
            'heartbeat_count': 45,
            'type': 'Factory'
        }
    },
    'datasite_summary': {
        'total': 8,
        'online': 6,
        'offline': 2
    },
    'experiment_status': {
        'current_experiment': 'PDM_CNN_Automotive_Experiment_001',
        'current_round': 7,
        'total_rounds': 10,
        'completed_experiments': 15,
        'total_experiments': 48,
        'failed_experiments': 1,
        'start_time': '2025-09-05T14:00:00.000000',
        'current_run': 2,
        'total_runs': 3
    },
    'timestamp': '2025-09-05T14:30:15.123456'
}
```

### **Advanced Dashboard Visualizations**

#### **Experiment Progress Visualization**
```html
<!-- Dashboard provides rich visualizations including: -->

<!-- 1. Circular Progress Ring for Round Progress -->
<div class="progress-ring">
    <svg>
        <circle class="bg" cx="40" cy="40" r="30"></circle>
        <circle class="progress" cx="40" cy="40" r="30" 
                stroke-dasharray="70% 30%"></circle>
    </svg>
    <div class="progress-text">70%</div>
</div>

<!-- 2. Linear Progress Bar for Overall Completion -->
<div class="progress-bar-container">
    <div class="progress-bar" style="width: 31.25%"></div>
</div>

<!-- 3. Status Indicators with Animation -->
<div class="status-dot online"></div> <!-- Animated pulse for online -->
<div class="status-dot offline"></div> <!-- Static red for offline -->

<!-- 4. Real-time Metrics Cards -->
<div class="experiment-card current">
    <div class="card-header">
        <span class="card-icon">🔬</span>
        <span class="card-title">Current Experiment</span>
    </div>
    <!-- Dynamic content with real-time updates -->
</div>
```

#### **Datasite Status Display**
```javascript
// JavaScript dashboard update functions
function updateDatasiteStatus(datasites, summary) {
    // Sort datasites for consistent display
    const sortedDatasites = Object.entries(datasites).sort(([a], [b]) => a.localeCompare(b));
    
    for (const [name, status] of sortedDatasites) {
        const isOnline = status.status === 'online';
        
        // Calculate time since last heartbeat
        const lastSeen = new Date(status.last_heartbeat);
        const diffMinutes = Math.floor((now - lastSeen) / (1000 * 60));
        
        // Display datasite with appropriate styling
        const datasite_html = `
            <div class="datasite ${isOnline ? 'online' : 'offline'}">
                <div class="status-dot ${isOnline ? 'online' : 'offline'}"></div>
                <div class="datasite-info">
                    <div class="datasite-name">${name}</div>
                    <div class="datasite-details">
                        <div>Status: ${isOnline ? 'Connected' : 'Disconnected'}</div>
                        <div>Last seen: ${formatTimeSince(diffMinutes)}</div>
                    </div>
                </div>
            </div>
        `;
    }
}
```

---

## 5. Integration with Federated Learning

### **End-to-End Monitoring Integration**

#### **Complete Federated Learning Monitoring Setup**
```python
def setup_comprehensive_monitoring(experiment_config):
    """Setup comprehensive monitoring for federated learning experiment."""
    
    # Initialize all monitoring components
    monitoring_setup = {
        'metrics_collector': MetricsCollector(),
        'heartbeat_manager': HeartbeatManager(port=8888, heartbeat_timeout=90),
        'dashboard': SimpleStatusDashboard(port=8889)
    }
    
    # Start monitoring services
    monitoring_setup['heartbeat_manager'].start()
    monitoring_setup['dashboard'].start()
    
    print("📊 Comprehensive monitoring started:")
    print(f"  - Heartbeat monitoring: Port 8888")
    print(f"  - Web dashboard: http://localhost:8889")
    print(f"  - Metrics collection: Active")
    
    return monitoring_setup

def execute_monitored_federated_learning(experiment_config, monitoring_setup):
    """Execute federated learning with comprehensive monitoring."""
    
    metrics_collector = monitoring_setup['metrics_collector']
    heartbeat_manager = monitoring_setup['heartbeat_manager']
    dashboard = monitoring_setup['dashboard']
    
    # Initialize experiment tracking
    dashboard.update_experiment_status(
        current_experiment=experiment_config['name'],
        total_rounds=experiment_config['max_rounds'],
        start_time=datetime.now().isoformat()
    )
    
    experiment_results = {
        'round_metrics': [],
        'overall_performance': {},
        'monitoring_summary': {}
    }
    
    # Execute federated rounds with monitoring
    for round_num in range(1, experiment_config['max_rounds'] + 1):
        print(f"🔄 Starting monitored federated round {round_num}")
        
        # Update dashboard
        dashboard.update_experiment_status(current_round=round_num)
        
        # Check datasite availability
        available_datasites = heartbeat_manager.get_available_datasites(
            experiment_config['required_datasites']
        )
        
        if len(available_datasites) < experiment_config['min_participants']:
            print(f"⚠️ Insufficient datasites for round {round_num}")
            continue
        
        # Execute federated round
        global_model, client_updates = execute_federated_round(
            round_num=round_num,
            available_datasites=available_datasites,
            experiment_config=experiment_config
        )
        
        # Collect comprehensive metrics
        round_metrics = metrics_collector.collect_round_metrics(
            round_num=round_num,
            global_model=global_model,
            client_updates=client_updates
        )
        
        experiment_results['round_metrics'].append(round_metrics)
        
        # Log round completion
        print(f"✅ Round {round_num} completed: "
              f"Accuracy={round_metrics.get('avg_accuracy', 0):.4f}, "
              f"Participants={len(available_datasites)}")
    
    # Collect final experiment metrics
    experiment_results['overall_performance'] = metrics_collector.collect_experiment_metrics(
        experiment_results
    )
    
    # Update dashboard with completion
    dashboard.update_experiment_status(
        status='completed',
        end_time=datetime.now().isoformat()
    )
    
    # Export comprehensive metrics
    metrics_collector.export_metrics(
        f"results/{experiment_config['name']}_monitoring_report.json"
    )
    
    return experiment_results
```

### **Monitoring Data Export and Analysis**

#### **Comprehensive Metrics Export**
```python
def export_monitoring_analytics(monitoring_setup, experiment_results, export_dir):
    """Export comprehensive monitoring analytics."""
    
    metrics_collector = monitoring_setup['metrics_collector']
    heartbeat_manager = monitoring_setup['heartbeat_manager']
    
    # Create export directory
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Export metrics data
    metrics_collector.export_metrics(export_path / "detailed_metrics.json")
    
    # Export datasite reliability report
    datasite_status = heartbeat_manager.get_all_datasite_status()
    reliability_report = generate_reliability_report(datasite_status)
    
    with open(export_path / "datasite_reliability.json", 'w') as f:
        json.dump(reliability_report, f, indent=2, default=str)
    
    # Export performance trends
    performance_trends = metrics_collector.get_performance_trends()
    
    with open(export_path / "performance_trends.json", 'w') as f:
        json.dump(performance_trends, f, indent=2)
    
    # Generate monitoring summary report
    monitoring_summary = {
        'experiment_overview': experiment_results['overall_performance'],
        'datasite_reliability': reliability_report,
        'performance_trends': performance_trends,
        'final_metrics': metrics_collector.get_final_metrics(),
        'export_timestamp': datetime.now().isoformat()
    }
    
    with open(export_path / "monitoring_summary.json", 'w') as f:
        json.dump(monitoring_summary, f, indent=2, default=str)
    
    print(f"📄 Monitoring analytics exported to: {export_path}")
    return monitoring_summary
```

---

## 6. Performance Analysis and Reporting

### **Advanced Analytics and Insights**

#### **Performance Trend Analysis**
```python
def analyze_federated_learning_performance(metrics_collector):
    """Analyze federated learning performance trends."""
    
    round_metrics = metrics_collector.round_metrics
    
    if len(round_metrics) < 3:
        return {'status': 'insufficient_data'}
    
    # Extract performance time series
    rounds = [rm['round_num'] for rm in round_metrics]
    accuracies = [rm.get('avg_accuracy', 0) for rm in round_metrics]
    losses = [rm.get('avg_loss', float('inf')) for rm in round_metrics]
    comm_times = [rm.get('total_communication_time', 0) for rm in round_metrics]
    training_times = [rm.get('total_training_time', 0) for rm in round_metrics]
    
    # Performance analysis
    performance_analysis = {
        'convergence_analysis': {
            'initial_accuracy': accuracies[0],
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'accuracy_improvement': accuracies[-1] - accuracies[0],
            'convergence_round': find_convergence_round(accuracies),
            'stability_score': calculate_stability_score(accuracies[-5:])
        },
        'efficiency_analysis': {
            'avg_comm_time_per_round': np.mean(comm_times),
            'avg_training_time_per_round': np.mean(training_times),
            'communication_efficiency_trend': analyze_efficiency_trend(comm_times),
            'training_efficiency_trend': analyze_efficiency_trend(training_times),
            'overall_efficiency_score': calculate_overall_efficiency(accuracies, comm_times, training_times)
        },
        'quality_analysis': {
            'performance_consistency': 1.0 / (np.var(accuracies) + 1e-8),
            'loss_reduction': losses[0] - losses[-1] if losses[0] != float('inf') else 0,
            'improvement_rate': calculate_improvement_rate(accuracies),
            'learning_curve_health': assess_learning_curve_health(accuracies, losses)
        }
    }
    
    return performance_analysis
```

### **Monitoring Best Practices**

#### **Production Monitoring Guidelines**
```python
def monitoring_best_practices_guide():
    """Guidelines for production monitoring deployment."""
    
    best_practices = {
        'metrics_collection': {
            'frequency': 'Every federated round',
            'storage': 'Persistent JSON export after each experiment',
            'retention': 'Keep metrics for compliance and analysis',
            'performance_impact': 'Minimal overhead (~1-2% of total time)'
        },
        'heartbeat_management': {
            'timeout_setting': '90 seconds (balance between responsiveness and network tolerance)',
            'check_frequency': '30 seconds background cleanup',
            'minimum_participants': 'At least 3 datasites for meaningful federated learning',
            'reliability_threshold': '70% uptime required for experiment continuation'
        },
        'dashboard_deployment': {
            'accessibility': 'Internal network only for security',
            'update_frequency': '10 seconds auto-refresh for real-time monitoring',
            'concurrent_users': 'Supports multiple simultaneous viewers',
            'mobile_compatibility': 'Responsive design for mobile monitoring'
        },
        'resource_management': {
            'memory_usage': 'Monitoring uses ~50MB RAM for typical experiments',
            'cpu_impact': 'Background threads use <5% CPU',
            'network_overhead': 'Heartbeats: ~1KB every 60 seconds per datasite',
            'storage_requirements': '~10MB per 100-round experiment'
        }
    }
    
    return best_practices
```

---

## 7. Troubleshooting and Diagnostics

### **Common Monitoring Issues**

#### **Heartbeat Connection Problems**
```python
def diagnose_heartbeat_issues(heartbeat_manager):
    """Diagnose common heartbeat connectivity issues."""
    
    diagnostic_results = {
        'port_accessibility': check_port_accessibility(heartbeat_manager.port),
        'firewall_status': check_firewall_configuration(heartbeat_manager.port),
        'datasite_connectivity': test_datasite_connectivity(),
        'network_latency': measure_network_latency(),
        'recommendations': []
    }
    
    # Generate recommendations based on diagnostics
    if not diagnostic_results['port_accessibility']:
        diagnostic_results['recommendations'].append(
            f"Port {heartbeat_manager.port} is not accessible. Check firewall settings."
        )
    
    if diagnostic_results['network_latency'] > 5000:  # >5 seconds
        diagnostic_results['recommendations'].append(
            "High network latency detected. Consider increasing heartbeat timeout."
        )
    
    return diagnostic_results

def troubleshoot_dashboard_access():
    """Troubleshoot dashboard accessibility issues."""
    
    troubleshooting_steps = [
        "1. Verify dashboard is running: Check for 'Status dashboard available' message",
        "2. Test local access: Open http://localhost:8889 in browser",
        "3. Check firewall: Ensure port 8889 is open for internal network access",
        "4. Verify network connectivity: Test from other machines on the network",
        "5. Check browser compatibility: Use modern browser with JavaScript enabled",
        "6. Review console errors: Check browser developer tools for errors"
    ]
    
    return troubleshooting_steps
```

### **Performance Monitoring Diagnostics**

#### **Metrics Collection Validation**
```python
def validate_metrics_collection(metrics_collector):
    """Validate that metrics collection is working correctly."""
    
    validation_results = {
        'collection_status': 'unknown',
        'data_quality': {},
        'completeness': {},
        'issues': [],
        'recommendations': []
    }
    
    # Check if any metrics have been collected
    if not metrics_collector.round_metrics:
        validation_results['collection_status'] = 'no_data'
        validation_results['issues'].append('No round metrics collected')
        validation_results['recommendations'].append('Verify collect_round_metrics() is called during federated rounds')
        return validation_results
    
    # Validate data quality
    recent_metrics = metrics_collector.round_metrics[-1]
    
    required_fields = ['round_num', 'timestamp', 'avg_accuracy', 'avg_loss']
    missing_fields = [field for field in required_fields if field not in recent_metrics]
    
    if missing_fields:
        validation_results['issues'].append(f'Missing required fields: {missing_fields}')
    
    # Check for reasonable values
    accuracy = recent_metrics.get('avg_accuracy', -1)
    if accuracy < 0 or accuracy > 1:
        validation_results['issues'].append(f'Invalid accuracy value: {accuracy}')
    
    loss = recent_metrics.get('avg_loss', -1)
    if loss < 0:
        validation_results['issues'].append(f'Invalid loss value: {loss}')
    
    validation_results['collection_status'] = 'healthy' if not validation_results['issues'] else 'issues_detected'
    
    return validation_results
```

---

**Developed by**: PDM-FDL Research Team  
**Last Updated**: September 2025  
**Monitoring Version**: 2.0  
**Dashboard Technology**: Flask + HTML5 + JavaScript  
**Real-time Features**: Heartbeat monitoring, live status updates, performance tracking  
**Production Ready**: Comprehensive monitoring for industrial federated learning networks
