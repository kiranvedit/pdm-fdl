"""
Data Integration Layer for Federated Learning
=============================================
Connects federated learning architecture with existing processed data.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Tuple, Any, Optional
import sys

# Add shared directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))
from utils.step1_data_utils import DataLoader as ProjectDataLoader


class FederatedDataDistributor:
    """
    Distributes processed data across federated learning datasites.
    Integrates with existing processed data structure.
    """
    
    def __init__(self, processed_data_path: str):
        self.processed_data_path = processed_data_path
        self.metadata = None
        self.data_loader = ProjectDataLoader(processed_data_path)
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from processed data."""
        # The ProjectDataLoader already loads metadata in its constructor
        self.metadata = self.data_loader.metadata
    
    def create_datasite_datasets(self, 
                                num_datasites: int, 
                                distribution_strategy: str = 'iid',
                                data_type: str = 'tabular',
                                alpha: float = 0.5) -> Dict[str, 'FederatedDataset']:
        """
        Create datasets for each datasite with specified distribution strategy.
        
        Args:
            num_datasites: Number of datasites to create
            distribution_strategy: 'iid', 'non_iid_label', 'non_iid_quantity'
            data_type: 'tabular', 'sequences', or 'multiclass'
            alpha: Dirichlet alpha for non-IID distribution
        """
        # Load appropriate data based on type
        if data_type == 'tabular':
            data_path = os.path.join(self.processed_data_path, 'tabular')
            X_train, y_train, X_val, y_val, X_test, y_test = self._load_tabular_data(data_path)
        elif data_type == 'sequences':
            data_path = os.path.join(self.processed_data_path, 'sequences')
            X_train, y_train, X_val, y_val, X_test, y_test = self._load_sequence_data(data_path)
        elif data_type == 'multiclass':
            data_path = os.path.join(self.processed_data_path, 'multiclass')
            X_train, y_train, X_val, y_val, X_test, y_test = self._load_multiclass_data(data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Combine train and validation data for distribution
        if X_val.size > 0 and y_val.size > 0:
            X_train_val = np.concatenate([X_train, X_val], axis=0)
            y_train_val = np.concatenate([y_train, y_val], axis=0)
        else:
            X_train_val = X_train
            y_train_val = y_train
        
        # Calculate split ratio for train/validation
        total_train_val_samples = len(X_train_val)
        train_ratio = len(X_train) / total_train_val_samples if total_train_val_samples > 0 else 0.8
        
        # Distribute combined train+validation data across datasites
        if distribution_strategy == 'iid':
            distributed_data = self._distribute_iid(X_train_val, y_train_val, num_datasites)
        elif distribution_strategy == 'non_iid_label':
            distributed_data = self._distribute_non_iid_label(X_train_val, y_train_val, num_datasites, alpha)
        elif distribution_strategy == 'non_iid_quantity':
            distributed_data = self._distribute_non_iid_quantity(X_train_val, y_train_val, num_datasites, alpha)
        elif distribution_strategy == 'non-iid':
            # Use Dirichlet distribution for non-IID data distribution
            distributed_data = self._distribute_non_iid_label(X_train_val, y_train_val, num_datasites, alpha)
        else:
            raise ValueError(f"Unsupported distribution strategy: {distribution_strategy}")
        
        # Split the distributed data back into train and validation for each datasite
        datasite_datasets = {}
        for datasite_id, (X_distributed, y_distributed) in distributed_data.items():
            # Calculate split point for this datasite
            n_samples = len(X_distributed)
            n_train = int(n_samples * train_ratio)
            
            # Split into train and validation
            X_train_split = X_distributed[:n_train]
            y_train_split = y_distributed[:n_train]
            X_val_split = X_distributed[n_train:]
            y_val_split = y_distributed[n_train:]
            
            # Create metadata with all data splits
            metadata = {
                'datasite_id': datasite_id,
                'distribution_strategy': distribution_strategy,
                'alpha': alpha,
                'data_type': data_type,
                'train_samples': len(X_train_split),
                'val_samples': len(X_val_split), 
                'test_samples': len(X_test),
                'features': list(range(X_train_split.shape[1])) if len(X_train_split.shape) > 1 else [0],
                'classes': list(np.unique(y_train_split)),
                # Store validation and test data in metadata
                'X_val': X_val_split,
                'y_val': y_val_split,
                'X_test': X_test,  # Complete test data (same for all)
                'y_test': y_test   # Complete test labels (same for all)
            }
            
            # Create FederatedDataset with training data and metadata containing val/test
            datasite_datasets[datasite_id] = FederatedDataset(
                X=X_train_split,  # Distributed training data
                y=y_train_split,  # Distributed training labels
                metadata=metadata,
                datasite_id=datasite_id
            )
        
        return datasite_datasets
    
    def _load_tabular_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load tabular data for federated learning."""
        train_file = os.path.join(data_path, 'train_features.npy')
        train_labels_file = os.path.join(data_path, 'train_labels.npy')
        test_file = os.path.join(data_path, 'test_features.npy')
        test_labels_file = os.path.join(data_path, 'test_labels.npy')
        
        if not all(os.path.exists(f) for f in [train_file, train_labels_file, test_file, test_labels_file]):
            # Use data loader to load and process data (it already knows where to find the data)
            data_dict = self.data_loader.load_tabular_data()
            # The DataLoader returns a dict with keys based on filenames (without .csv)
            # Expected keys: X_train, y_train, X_test, y_test, X_val, y_val
            
            X_train = data_dict.get('X_train')
            y_train = data_dict.get('y_train')
            X_test = data_dict.get('X_test')
            y_test = data_dict.get('y_test')
            X_val = data_dict.get('X_val', np.array([]))  # Default to empty if no validation data
            y_val = data_dict.get('y_val', np.array([]))
            
            if X_train is None or y_train is None or X_test is None or y_test is None:
                available_keys = list(data_dict.keys())
                raise ValueError(f"Could not find required data splits. Available keys: {available_keys}")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
        
        X_train = np.load(train_file)
        y_train = np.load(train_labels_file)
        X_test = np.load(test_file)
        y_test = np.load(test_labels_file)
        
        # Try to load validation data if available
        val_file = os.path.join(data_path, 'val_features.npy')
        val_labels_file = os.path.join(data_path, 'val_labels.npy')
        
        if os.path.exists(val_file) and os.path.exists(val_labels_file):
            X_val = np.load(val_file)
            y_val = np.load(val_labels_file)
        else:
            X_val = np.array([])
            y_val = np.array([])
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _load_sequence_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load sequence data for temporal models."""
        # Implementation for sequence data loading
        train_sequences = os.path.join(data_path, 'train_sequences.npy')
        train_labels = os.path.join(data_path, 'train_labels.npy')
        test_sequences = os.path.join(data_path, 'test_sequences.npy')
        test_labels = os.path.join(data_path, 'test_labels.npy')
        
        if all(os.path.exists(f) for f in [train_sequences, train_labels, test_sequences, test_labels]):
            X_train = np.load(train_sequences)
            y_train = np.load(train_labels)
            X_test = np.load(test_sequences)
            y_test = np.load(test_labels)
            
            # Try to load validation data if available
            val_sequences = os.path.join(data_path, 'val_sequences.npy')
            val_labels = os.path.join(data_path, 'val_labels.npy')
            
            if os.path.exists(val_sequences) and os.path.exists(val_labels):
                X_val = np.load(val_sequences)
                y_val = np.load(val_labels)
            else:
                X_val = np.array([])
                y_val = np.array([])
        else:
            # Fallback to tabular data if sequences not available
            return self._load_tabular_data(os.path.join(self.processed_data_path, 'tabular'))
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _load_multiclass_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load multiclass classification data."""
        # Implementation for multiclass data loading
        train_features = os.path.join(data_path, 'train_features.npy')
        train_labels = os.path.join(data_path, 'train_multiclass_labels.npy')
        test_features = os.path.join(data_path, 'test_features.npy')
        test_labels = os.path.join(data_path, 'test_multiclass_labels.npy')
        
        if all(os.path.exists(f) for f in [train_features, train_labels, test_features, test_labels]):
            X_train = np.load(train_features)
            y_train = np.load(train_labels)
            X_test = np.load(test_features)
            y_test = np.load(test_labels)
            
            # Try to load validation data if available
            val_features = os.path.join(data_path, 'val_features.npy')
            val_labels = os.path.join(data_path, 'val_multiclass_labels.npy')
            
            if os.path.exists(val_features) and os.path.exists(val_labels):
                X_val = np.load(val_features)
                y_val = np.load(val_labels)
            else:
                X_val = np.array([])
                y_val = np.array([])
        else:
            # Fallback to tabular data
            return self._load_tabular_data(os.path.join(self.processed_data_path, 'tabular'))
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _distribute_iid(self, X: np.ndarray, y: np.ndarray, num_datasites: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data in IID manner across datasites."""
        datasite_data = {}
        samples_per_site = len(X) // num_datasites
        
        # Shuffle data for random distribution
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(num_datasites):
            start_idx = i * samples_per_site
            if i == num_datasites - 1:  # Last datasite gets remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * samples_per_site
            
            site_X = X_shuffled[start_idx:end_idx]
            site_y = y_shuffled[start_idx:end_idx]
            
            datasite_id = f"datasite_{i+1}"
            datasite_data[datasite_id] = (site_X, site_y)
        
        return datasite_data
    
    def _distribute_non_iid_label(self, X: np.ndarray, y: np.ndarray, num_datasites: int, alpha: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data with label skew using Dirichlet distribution."""
        datasite_data = {}
        # Ensure y is 1D for proper boolean indexing
        if len(y.shape) > 1:
            y = y.flatten()
        
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)
        
        # Create Dirichlet distribution for each class
        class_distributions = np.random.dirichlet([alpha] * num_datasites, num_classes)
        
        # Initialize datasite data holders
        datasite_data_temp = {f"datasite_{i+1}": {'X': [], 'y': []} for i in range(num_datasites)}
        
        # Distribute each class according to Dirichlet distribution
        for class_idx, label in enumerate(unique_labels):
            class_mask = (y == label)  # Now this will be 1D boolean array
            
            # Handle different data dimensions properly
            if len(X.shape) == 2:  # 2D tabular data (samples, features)
                class_X = X[class_mask]
            elif len(X.shape) == 3:  # 3D sequence data (samples, timesteps, features)
                class_X = X[class_mask]
            else:
                raise ValueError(f"Unexpected X shape: {X.shape}")
                
            class_y = y[class_mask]
            
            class_distribution = class_distributions[class_idx]
            samples_per_site = (class_distribution * len(class_X)).astype(int)
            
            # Ensure all samples are distributed
            samples_per_site[-1] = len(class_X) - np.sum(samples_per_site[:-1])
            
            start_idx = 0
            for i, num_samples in enumerate(samples_per_site):
                if num_samples > 0:
                    end_idx = start_idx + num_samples
                    datasite_id = f"datasite_{i+1}"
                    datasite_data_temp[datasite_id]['X'].append(class_X[start_idx:end_idx])
                    datasite_data_temp[datasite_id]['y'].append(class_y[start_idx:end_idx])
                    start_idx = end_idx
        
        # Combine data for each datasite and return as tuples
        for datasite_id, data in datasite_data_temp.items():
            if data['X']:  # Only create dataset if datasite has data
                # Stack all arrays for this datasite
                site_X = np.vstack(data['X'])
                site_y = np.hstack(data['y'])
                
                # Shuffle datasite data
                indices = np.random.permutation(len(site_X))
                site_X = site_X[indices]
                site_y = site_y[indices]
                
                datasite_data[datasite_id] = (site_X, site_y)
        
        return datasite_data
        
        return datasite_datasets
    
    def _distribute_non_iid_quantity(self, X: np.ndarray, y: np.ndarray, num_datasites: int, alpha: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data with quantity skew using Dirichlet distribution."""
        # Generate quantity distribution using Dirichlet
        quantity_distribution = np.random.dirichlet([alpha] * num_datasites)
        samples_per_site = (quantity_distribution * len(X)).astype(int)
        
        # Ensure all samples are distributed
        samples_per_site[-1] = len(X) - np.sum(samples_per_site[:-1])
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        datasite_data = {}
        start_idx = 0
        
        for i, num_samples in enumerate(samples_per_site):
            if num_samples > 0:
                end_idx = start_idx + num_samples
                site_X = X_shuffled[start_idx:end_idx]
                site_y = y_shuffled[start_idx:end_idx]
                
                datasite_id = f"datasite_{i+1}"
                datasite_data[datasite_id] = (site_X, site_y)
                start_idx = end_idx
        
        return datasite_data
    
    def get_data_statistics(self, datasite_datasets: Dict[str, 'FederatedDataset']) -> Dict[str, Any]:
        """Get statistics about data distribution across datasites."""
        stats = {
            'num_datasites': len(datasite_datasets),
            'total_samples': 0,
            'samples_per_datasite': {},
            'label_distribution_per_datasite': {},
            'feature_statistics': {}
        }
        
        for datasite_id, dataset in datasite_datasets.items():
            num_samples = len(dataset)
            stats['total_samples'] += num_samples
            stats['samples_per_datasite'][datasite_id] = num_samples
            
            # Label distribution
            unique_labels, counts = np.unique(dataset.y, return_counts=True)
            stats['label_distribution_per_datasite'][datasite_id] = {
                str(label): int(count) for label, count in zip(unique_labels, counts)
            }
        
        return stats

    def get_complete_test_data(self, data_type: str) -> Dict[str, np.ndarray]:
        """
        Get complete test dataset for all datasites.
        
        Args:
            data_type: Type of data ('tabular', 'sequences', 'multiclass')
            
        Returns:
            Dictionary with complete test data (X_test, y_test)
        """
        # Load appropriate data based on type
        if data_type == 'tabular':
            data_path = os.path.join(self.processed_data_path, 'tabular')
            _, _, _, _, X_test, y_test = self._load_tabular_data(data_path)
        elif data_type == 'sequences':
            data_path = os.path.join(self.processed_data_path, 'sequences')
            _, _, _, _, X_test, y_test = self._load_sequence_data(data_path)
        elif data_type == 'multiclass':
            data_path = os.path.join(self.processed_data_path, 'multiclass')
            _, _, _, _, X_test, y_test = self._load_multiclass_data(data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return {
            'X_test': X_test,
            'y_test': y_test
        }


class FederatedDataset(Dataset):
    """
    PyTorch Dataset for federated learning with metadata integration.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, metadata: Dict[str, Any], datasite_id: Optional[str] = None):
        self.X = torch.FloatTensor(X)
        
        # Ensure y is 1D for proper loss function compatibility
        if len(y.shape) > 1:
            y = y.flatten()
        self.y = torch.LongTensor(y)
        
        self.metadata = metadata or {}
        self.datasite_id = datasite_id
        self.test_dataset = None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> TorchDataLoader:
        """Create PyTorch DataLoader for this dataset."""
        return TorchDataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from metadata."""
        return self.metadata.get('features', [])
    
    def get_target_names(self) -> List[str]:
        """Get target class names from metadata."""
        return self.metadata.get('target_classes', [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'num_features': self.X.shape[1] if len(self.X.shape) > 1 else 1,
            'num_classes': len(torch.unique(self.y)),
            'feature_means': torch.mean(self.X, dim=0).tolist(),
            'feature_stds': torch.std(self.X, dim=0).tolist(),
            'class_distribution': {
                str(label.item()): int(count.item()) 
                for label, count in zip(*torch.unique(self.y, return_counts=True))
            }
        }


class DataSiteDataManager:
    """
    Manages data for a specific datasite in federated learning.
    """
    
    def __init__(self, datasite_id: str, train_dataset: FederatedDataset):
        self.datasite_id = datasite_id
        self.train_dataset = train_dataset
        self.test_dataset = train_dataset.test_dataset
    
    def get_training_data(self, batch_size: int = 32) -> TorchDataLoader:
        """Get training data loader for this datasite."""
        return self.train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    
    def get_test_data(self, batch_size: int = 32) -> TorchDataLoader:
        """Get test data loader for evaluation."""
        if self.test_dataset is None:
            raise ValueError(f"No test dataset available for datasite {self.datasite_id}")
        return self.test_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about this datasite's data."""
        info = {
            'datasite_id': self.datasite_id,
            'train_statistics': self.train_dataset.get_statistics(),
            'feature_names': self.train_dataset.get_feature_names(),
            'target_names': self.train_dataset.get_target_names()
        }
        
        if self.test_dataset:
            info['test_statistics'] = self.test_dataset.get_statistics()
        
        return info
    
    def sample_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a subset of training data."""
        if num_samples >= len(self.train_dataset):
            return self.train_dataset.X, self.train_dataset.y
        
        indices = torch.randperm(len(self.train_dataset))[:num_samples]
        return self.train_dataset.X[indices], self.train_dataset.y[indices]
