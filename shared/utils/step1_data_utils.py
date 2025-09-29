#!/usr/bin/env python3
"""
Data Utilities for Federated Learning Predictive Maintenance
Provides utilities for loading, validating, and manipulating processed data
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Utility class for loading processed data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load metadata from processed data"""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def load_tabular_data(self) -> Dict[str, np.ndarray]:
        """Load tabular data from CSV files"""
        tabular_dir = os.path.join(self.data_dir, "tabular")
        
        if not os.path.exists(tabular_dir):
            raise FileNotFoundError(f"Tabular data directory not found: {tabular_dir}")
        
        data = {}
        for file_name in os.listdir(tabular_dir):
            if file_name.endswith('.csv'):
                key = file_name.replace('.csv', '')
                file_path = os.path.join(tabular_dir, file_name)
                
                df = pd.read_csv(file_path)
                data[key] = df.values
        
        logger.info(f"Loaded tabular data with keys: {list(data.keys())}")
        return data
    
    def load_sequence_data(self) -> Dict[str, np.ndarray]:
        """Load sequence data from CSV files for LSTM models"""
        sequence_dir = os.path.join(self.data_dir, "sequences")
        
        if not os.path.exists(sequence_dir):
            logger.warning("Sequence data directory not found")
            return {}
        
        data = {}
        for file_name in os.listdir(sequence_dir):
            if file_name.endswith('_sequences.csv'):
                key = file_name.replace('_sequences.csv', '')
                file_path = os.path.join(sequence_dir, file_name)
                
                df = pd.read_csv(file_path)
                
                # Reconstruct 3D array from CSV format
                # CSV format: sample_id, timestep, feature1, feature2, ...
                sample_ids = df['sample_id'].unique()
                timesteps = df['timestep'].unique()
                features = [col for col in df.columns if col not in ['sample_id', 'timestep']]
                
                # Create 3D array: (samples, timesteps, features)
                array_3d = np.zeros((len(sample_ids), len(timesteps), len(features)))
                
                for i, sample_id in enumerate(sample_ids):
                    sample_data = df[df['sample_id'] == sample_id].sort_values('timestep')
                    array_3d[i] = sample_data[features].values
                
                data[key] = array_3d
            elif file_name.endswith('.csv') and 'y_' in file_name:
                # Load target data for sequences
                key = file_name.replace('.csv', '')
                file_path = os.path.join(sequence_dir, file_name)
                
                df = pd.read_csv(file_path)
                data[key] = df.values.flatten()
        
        logger.info(f"Loaded sequence data with keys: {list(data.keys())}")
        return data
    
    def load_federated_data(self, client_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, np.ndarray]]:
        """Load federated client data - DEPRECATED: Training split used for all clients"""
        logger.warning("Federated data loading is deprecated. Use training split for all federated clients.")
        return {}
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from metadata"""
        return self.metadata.get('feature_names', [])
    
    def get_data_info(self) -> Dict:
        """Get comprehensive data information"""
        return {
            'metadata': self.metadata,
            'data_directory': self.data_dir,
            'available_splits': self._get_available_splits()
        }
    
    def _get_available_splits(self) -> List[str]:
        """Get list of available data splits"""
        splits = []
        
        # Check for tabular data
        if os.path.exists(os.path.join(self.data_dir, "tabular")):
            splits.append("tabular")
        
        # Check for sequence data
        if os.path.exists(os.path.join(self.data_dir, "sequences")):
            splits.append("sequences")
        
        # Check for federated data
        if os.path.exists(os.path.join(self.data_dir, "federated")):
            splits.append("federated")
        
        # Check for multiclass data
        if os.path.exists(os.path.join(self.data_dir, "multiclass")):
            splits.append("multiclass")
        
        return splits

class DataValidator:
    """Utility class for validating processed data"""
    
    @staticmethod
    def validate_data_splits(data: Dict[str, np.ndarray], 
                           expected_splits: List[str] = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']) -> bool:
        """Validate that data contains expected splits"""
        missing_splits = [split for split in expected_splits if split not in data]
        
        if missing_splits:
            logger.error(f"Missing data splits: {missing_splits}")
            return False
        
        # Check shapes consistency
        train_samples = len(data['X_train'])
        if len(data['y_train']) != train_samples:
            logger.error(f"Training data shape mismatch: X={data['X_train'].shape}, y={data['y_train'].shape}")
            return False
        
        val_samples = len(data['X_val'])
        if len(data['y_val']) != val_samples:
            logger.error(f"Validation data shape mismatch: X={data['X_val'].shape}, y={data['y_val'].shape}")
            return False
        
        test_samples = len(data['X_test'])
        if len(data['y_test']) != test_samples:
            logger.error(f"Test data shape mismatch: X={data['X_test'].shape}, y={data['y_test'].shape}")
            return False
        
        # Check feature dimensions consistency
        n_features_train = data['X_train'].shape[-1]
        n_features_val = data['X_val'].shape[-1]
        n_features_test = data['X_test'].shape[-1]
        
        if not (n_features_train == n_features_val == n_features_test):
            logger.error(f"Feature dimension mismatch: train={n_features_train}, val={n_features_val}, test={n_features_test}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    @staticmethod
    def validate_federated_data(federated_data: Dict[int, Dict[str, np.ndarray]]) -> bool:
        """Validate federated client data"""
        if not federated_data:
            logger.error("No federated data provided")
            return False
        
        # Check each client's data
        for client_id, client_data in federated_data.items():
            if 'X' not in client_data or 'y' not in client_data:
                logger.error(f"Client {client_id} missing X or y data")
                return False
            
            if len(client_data['X']) != len(client_data['y']):
                logger.error(f"Client {client_id} data shape mismatch: X={client_data['X'].shape}, y={client_data['y'].shape}")
                return False
        
        # Check feature consistency across clients
        feature_dims = [data['X'].shape[-1] for data in federated_data.values()]
        if len(set(feature_dims)) > 1:
            logger.error(f"Feature dimension mismatch across clients: {feature_dims}")
            return False
        
        logger.info(f"Federated data validation passed for {len(federated_data)} clients")
        return True
    
    @staticmethod
    def check_data_distribution(y: np.ndarray, name: str = "Data") -> Dict:
        """Check and report class distribution"""
        unique_classes, counts = np.unique(y, return_counts=True)
        percentages = counts / len(y) * 100
        
        distribution_info = {
            'classes': unique_classes.tolist(),
            'counts': counts.tolist(),
            'percentages': percentages.tolist(),
            'total_samples': len(y)
        }
        
        logger.info(f"{name} distribution:")
        for cls, count, pct in zip(unique_classes, counts, percentages):
            logger.info(f"  Class {cls}: {count} samples ({pct:.2f}%)")
        
        return distribution_info

class DataVisualizer:
    """Utility class for visualizing processed data"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_data_distribution(self, data: Dict[str, np.ndarray], 
                             title: str = "Data Distribution") -> None:
        """Plot class distribution across splits"""
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        splits = ['train', 'val', 'test']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        for i, split in enumerate(splits):
            y_key = f'y_{split}'
            if y_key in data:
                unique, counts = np.unique(data[y_key], return_counts=True)
                axes[i].bar(unique, counts, color=colors[i], alpha=0.7)
                axes[i].set_title(f'{split.capitalize()} Set')
                axes[i].set_xlabel('Class')
                axes[i].set_ylabel('Count')
                
                # Add percentage labels
                total = len(data[y_key])
                for j, (cls, count) in enumerate(zip(unique, counts)):
                    pct = count / total * 100
                    axes[i].text(cls, count + max(counts) * 0.01, f'{pct:.1f}%', 
                               ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, X: np.ndarray, feature_names: List[str],
                                 sample_size: int = 1000) -> None:
        """Plot feature distributions"""
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        n_features = X_sample.shape[-1]
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(n_features):
            if len(X_sample.shape) == 3:  # Sequence data
                feature_data = X_sample[:, -1, i]  # Use last timestep
            else:  # Tabular data
                feature_data = X_sample[:, i]
            
            axes[i].hist(feature_data, bins=30, alpha=0.7, color='skyblue')
            feature_name = feature_names[i] if i < len(feature_names) else f'Feature_{i}'
            axes[i].set_title(feature_name)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_federated_distribution(self, federated_data: Dict[int, Dict[str, np.ndarray]]) -> None:
        """Plot class distribution for each federated client"""
        n_clients = len(federated_data)
        n_cols = 4
        n_rows = (n_clients + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_clients))
        
        for i, (client_id, client_data) in enumerate(federated_data.items()):
            unique, counts = np.unique(client_data['y'], return_counts=True)
            
            axes[i].bar(unique, counts, color=colors[i], alpha=0.7)
            axes[i].set_title(f'Client {client_id}')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
            
            # Add sample count
            total_samples = len(client_data['y'])
            axes[i].text(0.5, 0.95, f'n={total_samples}', 
                        transform=axes[i].transAxes, ha='center', va='top')
        
        # Hide unused subplots
        for i in range(n_clients, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Federated Client Data Distribution')
        plt.tight_layout()
        plt.show()
    
    def plot_sequence_data_sample(self, X_seq: np.ndarray, y_seq: np.ndarray,
                                feature_names: List[str], n_samples: int = 3) -> None:
        """Plot sample sequences for LSTM data"""
        n_features = X_seq.shape[-1]
        n_samples = min(n_samples, len(X_seq))
        
        fig, axes = plt.subplots(n_samples, n_features, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            sample_idx = np.random.randint(0, len(X_seq))
            sequence = X_seq[sample_idx]
            label = y_seq[sample_idx]
            
            for j in range(n_features):
                axes[i, j].plot(sequence[:, j], marker='o', markersize=3)
                feature_name = feature_names[j] if j < len(feature_names) else f'Feature_{j}'
                axes[i, j].set_title(f'{feature_name} (Label: {label})')
                axes[i, j].set_xlabel('Time Step')
                axes[i, j].set_ylabel('Value')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.suptitle('Sample Sequences for LSTM Training')
        plt.tight_layout()
        plt.show()

# Convenience functions
def quick_load_data(data_dir: str, data_type: str = 'tabular') -> Tuple[Dict, Dict]:
    """Quickly load data and metadata"""
    loader = DataLoader(data_dir)
    
    if data_type == 'tabular':
        data = loader.load_tabular_data()
    elif data_type == 'sequences':
        data = loader.load_sequence_data()
    elif data_type == 'federated':
        data = loader.load_federated_data()
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    metadata = loader.metadata
    return data, metadata

def validate_and_visualize(data_dir: str, visualize: bool = True) -> bool:
    """Complete validation and visualization pipeline"""
    try:
        # Load data
        loader = DataLoader(data_dir)
        
        # Load and validate tabular data
        tabular_data = loader.load_tabular_data()
        is_valid = DataValidator.validate_data_splits(tabular_data)
        
        if not is_valid:
            return False
        
        # Check for federated data
        try:
            federated_data = loader.load_federated_data()
            DataValidator.validate_federated_data(federated_data)
        except FileNotFoundError:
            logger.info("No federated data found - skipping federated validation")
            federated_data = None
        
        if visualize:
            visualizer = DataVisualizer()
            
            # Plot data distribution
            visualizer.plot_data_distribution(tabular_data)
            
            # Plot feature distributions
            feature_names = loader.get_feature_names()
            visualizer.plot_feature_distributions(tabular_data['X_train'], feature_names)
            
            # Plot federated distribution if available
            if federated_data:
                visualizer.plot_federated_distribution(federated_data)
            
            # Plot sequence data if available
            try:
                sequence_data = loader.load_sequence_data()
                if sequence_data:
                    visualizer.plot_sequence_data_sample(
                        sequence_data['X_train'], 
                        sequence_data['y_train'], 
                        feature_names
                    )
            except:
                logger.info("No sequence data found - skipping sequence visualization")
        
        logger.info("Data validation and visualization completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in validation: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    data_dir = "/workspaces/pdm-fdl/shared/processed_data"
    
    if os.path.exists(data_dir):
        validate_and_visualize(data_dir, visualize=True)
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please run step1_data_preparation.py first")
