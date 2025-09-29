"""
Data Preparation and Distribution Functions for Federated Learning
Handles IID and Non-IID data distribution strategies
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional


def create_data_distribution(X_train, y_train, num_clients: int = 4, distribution_type: str = 'iid', 
                           alpha: float = 0.5, min_samples_per_client: Optional[int] = None) -> Dict[str, Tuple]:
    """
    Create federated data distribution among clients with comprehensive options
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_clients: Number of federated clients
        distribution_type: 'iid' or 'non_iid'
        alpha: Dirichlet concentration parameter for non-IID (lower = more heterogeneous)
        min_samples_per_client: Minimum samples guaranteed per client (auto-calculated if None)
        
    Returns:
        Dictionary mapping client_id to (X_client, y_client) data tuples
    """
    print(f"Creating {distribution_type.upper()} data distribution for {num_clients} clients...")
    
    n_samples = len(X_train)
    n_classes = len(np.unique(y_train))
    
    # Auto-calculate reasonable minimum samples per client
    if min_samples_per_client is None:
        # Ensure at least 32 samples per client (good for batch training) but not more than 20% of data per client
        min_samples_per_client = max(32, min(n_samples // (num_clients * 5), n_samples // num_clients))
    
    print(f"   Total samples: {n_samples}")
    print(f"   Classes: {n_classes}")
    print(f"   Min samples per client: {min_samples_per_client}")
    
    # Ensure we have enough samples
    if n_samples < num_clients * min_samples_per_client:
        # Adjust minimum to what's actually possible
        min_samples_per_client = max(10, n_samples // num_clients)
        print(f"   Adjusted min samples per client to: {min_samples_per_client}")
    
    client_data_map = {}
    
    if distribution_type == 'iid':
        # IID: Random uniform distribution
        print(f"   Creating IID distribution...")
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Calculate base samples per client
        base_samples_per_client = n_samples // num_clients
        remaining_samples = n_samples % num_clients
        
        # Ensure minimum samples requirement
        if base_samples_per_client < min_samples_per_client:
            print(f"   Adjusting distribution to meet minimum sample requirements...")
        
        current_idx = 0
        for client_id in range(num_clients):
            # Calculate samples for this client
            samples_for_client = max(min_samples_per_client, base_samples_per_client)
            if client_id < remaining_samples:
                samples_for_client += 1
            
            # Ensure we don't exceed available samples
            if current_idx + samples_for_client > n_samples:
                samples_for_client = n_samples - current_idx
            
            # Extract indices for this client
            end_idx = min(current_idx + samples_for_client, n_samples)
            client_indices = indices[current_idx:end_idx]
            current_idx = end_idx
            
            X_client = X_train[client_indices]
            y_client = y_train[client_indices]
            
            client_data_map[f"client_{client_id}"] = (X_client, y_client)
            
            # Print distribution info
            unique_classes, class_counts = np.unique(y_client, return_counts=True)
            class_dist = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
            print(f"   Client {client_id}: {len(X_client)} samples, classes: {class_dist}")
            
            # Break if we've used all samples
            if current_idx >= n_samples:
                break
    
    elif distribution_type == 'non_iid':
        # Non-IID: Dirichlet distribution for class heterogeneity with minimum sample guarantee
        print(f"   Creating Non-IID distribution (alpha={alpha})...")
        
        # Group samples by class
        class_indices = {}
        for class_label in range(n_classes):
            class_indices[class_label] = np.where(y_train == class_label)[0]
            print(f"   Class {class_label}: {len(class_indices[class_label])} samples")
        
        # Initialize client allocations
        client_allocations = {f"client_{i}": [] for i in range(num_clients)}
        
        # First pass: Ensure minimum samples per client using round-robin
        print(f"   First pass: Ensuring minimum {min_samples_per_client} samples per client...")
        
        # Create a pool of all indices
        all_indices = list(range(n_samples))
        np.random.shuffle(all_indices)
        
        allocated_indices = set()
        
        # Distribute minimum samples to each client
        idx = 0
        for client_id in range(num_clients):
            client_key = f"client_{client_id}"
            samples_needed = min_samples_per_client
            
            while samples_needed > 0 and idx < len(all_indices):
                if all_indices[idx] not in allocated_indices:
                    client_allocations[client_key].append(all_indices[idx])
                    allocated_indices.add(all_indices[idx])
                    samples_needed -= 1
                idx += 1
        
        # Second pass: Distribute remaining samples using Dirichlet distribution
        remaining_indices = [i for i in all_indices if i not in allocated_indices]
        
        if remaining_indices:
            print(f"   Second pass: Distributing {len(remaining_indices)} remaining samples with Dirichlet...")
            
            # Group remaining indices by class
            remaining_by_class = {}
            for idx in remaining_indices:
                class_label = y_train[idx]
                if class_label not in remaining_by_class:
                    remaining_by_class[class_label] = []
                remaining_by_class[class_label].append(idx)
            
            # For each class, use Dirichlet to distribute among clients
            for class_label, class_remaining_indices in remaining_by_class.items():
                if len(class_remaining_indices) == 0:
                    continue
                    
                # Generate Dirichlet proportions for this class
                dirichlet_params = np.repeat(alpha, num_clients)
                proportions = np.random.dirichlet(dirichlet_params)
                
                # Shuffle class indices
                np.random.shuffle(class_remaining_indices)
                
                # Distribute according to proportions
                current_idx = 0
                for client_id in range(num_clients):
                    client_key = f"client_{client_id}"
                    n_samples_for_client = int(len(class_remaining_indices) * proportions[client_id])
                    
                    # Take samples for this client
                    end_idx = min(current_idx + n_samples_for_client, len(class_remaining_indices))
                    selected_indices = class_remaining_indices[current_idx:end_idx]
                    client_allocations[client_key].extend(selected_indices)
                    current_idx = end_idx
                
                # Distribute any remaining samples
                while current_idx < len(class_remaining_indices):
                    client_id = current_idx % num_clients
                    client_key = f"client_{client_id}"
                    client_allocations[client_key].append(class_remaining_indices[current_idx])
                    current_idx += 1
        
        # Create final client data
        for client_id in range(num_clients):
            client_key = f"client_{client_id}"
            client_indices = np.array(client_allocations[client_key])
            
            # Shuffle to mix classes
            np.random.shuffle(client_indices)
            
            X_client = X_train[client_indices]
            y_client = y_train[client_indices]
            
            client_data_map[client_key] = (X_client, y_client)
            
            # Print distribution info
            unique_classes, class_counts = np.unique(y_client, return_counts=True)
            class_dist = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
            print(f"   Client {client_id}: {len(X_client)} samples, classes: {class_dist}")
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return client_data_map


def create_federated_clients(client_data_map: Dict[str, Tuple], model_class, model_params: Dict,
                           device: str = 'cpu', local_epochs: int = 5, batch_size: int = 16,
                           learning_rate: float = 0.001) -> Dict[str, Dict]:
    """
    Create federated client data configurations from data distribution map
    Compatible with ClientManager architecture
    
    Args:
        client_data_map: Dictionary mapping client_id to (X_client, y_client)
        model_class: Model class for each client
        model_params: Model parameters
        device: Training device
        local_epochs: Local training epochs per round
        batch_size: Training batch size
        learning_rate: Learning rate for local training
        
    Returns:
        Dictionary of client configurations ready for ClientManager initialization
    """
    client_configs = {}
    
    print(f"Creating {len(client_data_map)} federated client configurations...")
    print(f"   Local epochs: {local_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    
    for client_id, (X_client, y_client) in client_data_map.items():
        # Preprocess data for LSTM models
        if 'LSTM' in model_class.__name__:
            # LSTM expects 3D input: (batch_size, sequence_length, features)
            if len(X_client.shape) == 2:
                # Reshape from (samples, features) to (samples, 1, features)
                # This treats each sample as a sequence of length 1
                X_client = X_client.reshape(X_client.shape[0], 1, X_client.shape[1])
                print(f"   {client_id}: Reshaped data from {(X_client.shape[0], X_client.shape[2])} to {X_client.shape} for LSTM")
        
        # Create configuration dictionary for ClientManager
        client_config = {
            'client_id': client_id,
            'local_data': (X_client, y_client),
            'model_class': model_class,
            'model_params': model_params,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': device,
            'data_samples': len(X_client),
            'data_shape': X_client.shape
        }
        
        client_configs[client_id] = client_config
    
    print(f"   Successfully created {len(client_configs)} federated client configurations")
    print(f"   Ready for ClientManager initialization")
    
    return client_configs


def load_model_specific_data(model_type='tabular'):
    """
    Load data in the appropriate format for different model types:
    - CNN: tabular (2D) format
    - LSTM: sequences (3D) format  
    - Hybrid: sequences (3D) format
    
    Args:
        model_type: 'tabular' for CNN, 'sequences' for LSTM/Hybrid, 'multiclass' for multiclass labels
    
    Returns X_fed, X_test, y_fed, y_test, data_info
    """
    import os
    import sys
    import pandas as pd
    
    print(f"Loading {model_type} data for federated learning...")
    
    # Import step1_data_utils from local utils directory
    try:
        from .step1_data_utils import DataLoader
    except ImportError as e:
        print(f"ERROR: Could not import step1_data_utils: {e}")
        print("SOLUTION: Please ensure step1_data_utils.py is in utils/ directory")
        raise ImportError("step1_data_utils is required. No fallback mechanism provided.")
    
    try:
        # Initialize data loader with processed data directory
        if sys.platform.startswith('win'):  # Windows
            data_dir = r"D:\Development\pdm-fdl\Step1-Experiment\NetworkFed\processed_data"
        elif sys.platform.startswith('linux'): # Linux/Codespace
            data_dir = '/workspaces/pdm-fdl/shared/processed_data'
        else:
            data_dir = input("Please enter the full path to processed_data directory: ")

        loader = DataLoader(data_dir)
        
        print(f"   Loading from: {data_dir}")
        print(f"   Data format: {model_type}")
        
        # Load data based on model type
        if model_type == 'tabular':
            # For CNN models - use tabular data (2D)
            data = loader.load_tabular_data()
            X_train = data['X_train']
            y_train = data['y_train'].flatten()
            X_test = data['X_test'] 
            y_test = data['y_test'].flatten()
            print(f"   Loaded tabular data for CNN models")
            
        elif model_type == 'sequences':
            # For LSTM/Hybrid models - use sequences data (3D)
            try:
                data = loader.load_sequence_data()  # Corrected method name
                if 'X_train' in data and 'X_test' in data:
                    X_train = data['X_train']
                    X_test = data['X_test']
                else:
                    # Alternative naming
                    X_train = data.get('X_train_sequences', data.get('X_train', None))
                    X_test = data.get('X_test_sequences', data.get('X_test', None))
                
                y_train = data['y_train'].flatten()
                y_test = data['y_test'].flatten()
                print(f"   Loaded sequences data for LSTM/Hybrid models")
                
                if X_train is None or X_test is None:
                    raise ValueError("Sequence data not found in expected format")
                    
            except Exception as e:
                print(f"   WARNING: Could not load sequence data: {e}")
                print("   FALLBACK: Using tabular data and reshaping for sequence models")
                # Fallback to tabular data and reshape
                tabular_data = loader.load_tabular_data()
                X_train_2d = tabular_data['X_train']
                X_test_2d = tabular_data['X_test']
                
                # Reshape to 3D: (samples, 1, features) - treating each sample as sequence length 1
                X_train = X_train_2d.reshape(X_train_2d.shape[0], 1, X_train_2d.shape[1])
                X_test = X_test_2d.reshape(X_test_2d.shape[0], 1, X_test_2d.shape[1])
                
                y_train = tabular_data['y_train'].flatten()
                y_test = tabular_data['y_test'].flatten()
                
                print(f"   Reshaped tabular data: {X_train_2d.shape} -> {X_train.shape}")
            
        elif model_type == 'multiclass':
            # For multiclass scenarios - load multiclass labels with tabular features
            tabular_data = loader.load_tabular_data()
            X_train = tabular_data['X_train']
            X_test = tabular_data['X_test']
            
            # Try to load multiclass labels
            try:
                multiclass_dir = os.path.join(data_dir, "multiclass")
                y_train_df = pd.read_csv(os.path.join(multiclass_dir, "y_train.csv"))
                y_test_df = pd.read_csv(os.path.join(multiclass_dir, "y_test.csv"))
                y_train = y_train_df.values.flatten()
                y_test = y_test_df.values.flatten()
                print(f"   Loaded multiclass labels")
            except:
                print(f"   WARNING: Multiclass labels not found, using binary labels")
                y_train = tabular_data['y_train'].flatten()
                y_test = tabular_data['y_test'].flatten()
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'tabular', 'sequences', or 'multiclass'")
        
        print(f"   Training data: {X_train.shape}")
        print(f"   Test data: {X_test.shape}")
        print(f"   Training labels: {y_train.shape}")
        print(f"   Test labels: {y_test.shape}")
        
        # Convert to tensors
        X_fed = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_fed = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Get metadata and feature information
        metadata = loader.metadata
        
        # Create comprehensive data info
        data_info = {
            'num_features': X_train.shape[-1],  # Last dimension for both 2D and 3D
            'num_classes': len(np.unique(y_train)),
            'num_train_samples': len(X_fed),
            'num_test_samples': len(X_test_tensor),
            'data_format': model_type,
            'data_shape': X_train.shape,
            'feature_names': metadata.get('feature_names', []),
            'data_source': 'processed_industrial_iot',
            'normalization': metadata.get('config', {}).get('normalize_method', 'unknown'),
            'balanced': metadata.get('config', {}).get('handle_imbalance', False)
        }
        
        # For sequences, add sequence length info
        if model_type == 'sequences' and len(X_train.shape) == 3:
            data_info['sequence_length'] = X_train.shape[1]
            data_info['num_features'] = X_train.shape[2]  # Features per timestep
        
        print(f"Data loaded successfully:")
        print(f"   Training data: {X_fed.shape}")
        print(f"   Test data: {X_test_tensor.shape}")
        print(f"   Features: {data_info['num_features']}")
        print(f"   Classes: {data_info['num_classes']}")
        print(f"   Format: {data_info['data_format']}")
        if 'sequence_length' in data_info:
            print(f"   Sequence length: {data_info['sequence_length']}")
        print(f"   Source: Industrial IoT Predictive Maintenance Dataset")
        
        return X_fed, X_test_tensor, y_fed, y_test_tensor, data_info
        
    except Exception as e:
        print(f"ERROR: Could not load processed data: {e}")
        print("SOLUTION: Please run the following to prepare data:")
        print("   1. Navigate to shared/utils/")
        print("   2. Run: python step1_data_preparation.py")
        print("   3. Ensure processed_data directory exists with required files")
        print(f"   4. Check that {model_type} format is available")
        raise FileNotFoundError("Processed data is required. No fallback mechanism provided.")
