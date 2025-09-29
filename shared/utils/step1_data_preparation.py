#!/usr/bin/env python3
"""
Data Preparation Package for Federated Learning Predictive Maintenance
Implements comprehensive data preprocessing for CNN, LSTM, and Hybrid models
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration class for data preprocessing parameters"""
    # Data paths
    #data_path: str = "/workspaces/pdm-fdl/shared/data/ai4i2020.csv"
    #output_dir: str = "/workspaces/pdm-fdl/shared/processed_data"
    
    if sys.platform.startswith('win'):  # Checks if it's Windows
        data_path = "D:\\Development\\pdm-fdl\\shared\\data\\ai4i2020.csv"
        output_dir = "D:\\Development\\pdm-fdl\\shared\\processed_data"

    elif sys.platform.startswith('linux'): # Checks if it's Linux
        data_path: str = "/workspaces/pdm-fdl/shared/data/ai4i2020.csv"
        output_dir: str = "/workspaces/pdm-fdl/shared/processed_data"
    else:  # Handle other operating systems if needed
        data_path = "path/for/other/os" # Default or raise an error

    # Split ratios (60% train, 20% validation, 20% test)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Preprocessing parameters
    normalize_method: str = 'minmax'  # 'minmax', 'standard', 'robust'
    handle_imbalance: bool = True
    imbalance_method: str = 'smote'  # 'smote', 'smote_enn', 'class_weight'
    
    # Feature engineering
    create_sequences: bool = True
    sequence_length: int = 10
    overlap_ratio: float = 0.5
    
    # Target variables
    binary_classification: bool = True  # Machine failure (0/1)
    multiclass_classification: bool = True  # Failure types
    
    # Federated learning parameters
    num_clients: int = 10
    non_iid_alpha: float = 0.5  # Dirichlet distribution parameter
    
    # Random seed for reproducibility
    random_seed: int = 42

class DataPreprocessor:
    """Main data preprocessing class for federated learning predictive maintenance"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data = None
        self.features = None
        self.target_binary = None
        self.target_multiclass = None
        self.feature_names = None
        self.scalers = {}
        self.label_encoders = {}
        
        # Set random seeds
        np.random.seed(config.random_seed)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data validation"""
        logger.info(f"Loading data from {self.config.data_path}")
        
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        
        self.data = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded dataset with shape: {self.data.shape}")
        
        # Display basic info
        logger.info("Dataset Info:")
        logger.info(f"Columns: {list(self.data.columns)}")
        logger.info(f"Missing values: {self.data.isnull().sum().sum()}")
        logger.info(f"Duplicates: {self.data.duplicated().sum()}")
        
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the raw data"""
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        if self.data.isnull().sum().sum() > 0:
            logger.info("Handling missing values with forward fill")
            self.data = self.data.fillna(method='ffill')
        
        # Remove duplicates
        initial_shape = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        removed_duplicates = initial_shape - self.data.shape[0]
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        logger.info(f"Cleaned dataset shape: {self.data.shape}")
        return self.data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features based on domain knowledge"""
        logger.info("Engineering features...")
        
        # Create temperature difference feature
        if 'Air temperature [K]' in self.data.columns and 'Process temperature [K]' in self.data.columns:
            self.data['Temperature_diff'] = (self.data['Process temperature [K]'] - 
                                           self.data['Air temperature [K]'])
        
        # Create power-related features
        if 'Torque [Nm]' in self.data.columns and 'Rotational speed [rpm]' in self.data.columns:
            # Power = Torque × Angular velocity
            # Angular velocity = RPM × 2π / 60
            self.data['Power_estimate'] = (self.data['Torque [Nm]'] * 
                                         self.data['Rotational speed [rpm]'] * 2 * np.pi / 60)
        
        # Create stress indicators
        if 'Tool wear [min]' in self.data.columns:
            self.data['Tool_wear_normalized'] = self.data['Tool wear [min]'] / self.data['Tool wear [min]'].max()
        
        # Create efficiency ratios
        if 'Torque [Nm]' in self.data.columns and 'Rotational speed [rpm]' in self.data.columns:
            self.data['Torque_speed_ratio'] = self.data['Torque [Nm]'] / (self.data['Rotational speed [rpm]'] + 1e-8)
        
        logger.info(f"Feature engineering complete. New shape: {self.data.shape}")
        return self.data
    
    def prepare_features_and_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vectors"""
        logger.info("Preparing features and targets...")
        
        # Identify feature columns (exclude ID, target, and non-numeric columns)
        exclude_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Get numerical features
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Handle categorical features (Type)
        if 'Type' in self.data.columns:
            le_type = LabelEncoder()
            self.data['Type_encoded'] = le_type.fit_transform(self.data['Type'])
            feature_cols.append('Type_encoded')
            self.label_encoders['Type'] = le_type
        
        # Prepare features
        self.features = self.data[feature_cols].values
        self.feature_names = feature_cols
        
        # Prepare targets
        if self.config.binary_classification:
            self.target_binary = self.data['Machine failure'].values
        
        if self.config.multiclass_classification:
            # Create multiclass target from failure modes
            failure_modes = []
            for idx, row in self.data.iterrows():
                if row['Machine failure'] == 0:
                    failure_modes.append('No_Failure')
                elif row.get('HDF', 0) == 1:
                    failure_modes.append('Heat_Dissipation_Failure')
                elif row.get('PWF', 0) == 1:
                    failure_modes.append('Power_Failure')
                elif row.get('OSF', 0) == 1:
                    failure_modes.append('Overstrain_Failure')
                elif row.get('TWF', 0) == 1:
                    failure_modes.append('Tool_Wear_Failure')
                elif row.get('RNF', 0) == 1:
                    failure_modes.append('Random_Failure')
                else:
                    failure_modes.append('Unknown_Failure')
            
            le_multiclass = LabelEncoder()
            self.target_multiclass = le_multiclass.fit_transform(failure_modes)
            self.label_encoders['multiclass'] = le_multiclass
        
        logger.info(f"Features prepared: {self.features.shape}")
        logger.info(f"Feature names: {self.feature_names}")
        if self.target_binary is not None:
            logger.info(f"Binary target distribution: {np.bincount(self.target_binary)}")
        if self.target_multiclass is not None:
            logger.info(f"Multiclass target distribution: {np.bincount(self.target_multiclass)}")
        
        return self.features, self.target_binary, self.target_multiclass
    
    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                          X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features using specified method"""
        logger.info(f"Normalizing features using {self.config.normalize_method} method")
        
        if self.config.normalize_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.config.normalize_method == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown normalization method: {self.config.normalize_method}")
            scaler = StandardScaler()
        
        # Fit on training data only
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['features'] = scaler
        
        logger.info("Feature normalization complete")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray, 
                             target_type: str = 'binary') -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using specified method"""
        if not self.config.handle_imbalance:
            return X_train, y_train
        
        logger.info(f"Handling class imbalance for {target_type} target using {self.config.imbalance_method}")
        
        original_distribution = np.bincount(y_train)
        logger.info(f"Original class distribution: {original_distribution}")
        
        if self.config.imbalance_method == 'smote':
            smote = SMOTE(random_state=self.config.random_seed)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        elif self.config.imbalance_method == 'smote_enn':
            smote_enn = SMOTEENN(random_state=self.config.random_seed)
            X_balanced, y_balanced = smote_enn.fit_resample(X_train, y_train)
        else:
            logger.warning(f"Unknown imbalance method: {self.config.imbalance_method}")
            return X_train, y_train
        
        balanced_distribution = np.bincount(y_balanced)
        logger.info(f"Balanced class distribution: {balanced_distribution}")
        
        return X_balanced, y_balanced
    
    def create_sequences_for_lstm(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        if not self.config.create_sequences:
            return X, y
        
        logger.info(f"Creating sequences of length {self.config.sequence_length}")
        
        X_sequences = []
        y_sequences = []
        
        step_size = int(self.config.sequence_length * (1 - self.config.overlap_ratio))
        step_size = max(1, step_size)  # Ensure at least step size of 1
        
        for i in range(0, len(X) - self.config.sequence_length + 1, step_size):
            X_sequences.append(X[i:i + self.config.sequence_length])
            y_sequences.append(y[i + self.config.sequence_length - 1])  # Use last label in sequence
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def split_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Split data into train/validation/test sets"""
        logger.info(f"Splitting data into {self.config.train_ratio:.0%}/{self.config.val_ratio:.0%}/{self.config.test_ratio:.0%}")
        
        # First split: separate test set
        X_temp, X_test, y_bin_temp, y_bin_test = train_test_split(
            self.features, self.target_binary,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=self.target_binary
        )
        
        # Second split: separate train and validation from temp
        val_size_adjusted = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        X_train, X_val, y_bin_train, y_bin_val = train_test_split(
            X_temp, y_bin_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_seed,
            stratify=y_bin_temp
        )
        
        # Normalize features
        X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        # Handle class imbalance for binary classification
        if self.config.handle_imbalance:
            X_train, y_bin_train = self.handle_class_imbalance(X_train, y_bin_train, 'binary')
        
        # Prepare data splits dictionary
        data_splits = {
            'tabular': {
                'X_train': X_train, 'y_train': y_bin_train,
                'X_val': X_val, 'y_val': y_bin_val,
                'X_test': X_test, 'y_test': y_bin_test
            }
        }
        
        # Create sequences for LSTM if requested
        if self.config.create_sequences:
            X_train_seq, y_train_seq = self.create_sequences_for_lstm(X_train, y_bin_train)
            X_val_seq, y_val_seq = self.create_sequences_for_lstm(X_val, y_bin_val)
            X_test_seq, y_test_seq = self.create_sequences_for_lstm(X_test, y_bin_test)
            
            data_splits['sequences'] = {
                'X_train': X_train_seq, 'y_train': y_train_seq,
                'X_val': X_val_seq, 'y_val': y_val_seq,
                'X_test': X_test_seq, 'y_test': y_test_seq
            }
        
        # Handle multiclass targets if requested
        if self.config.multiclass_classification and self.target_multiclass is not None:
            # Split multiclass targets using same indices
            y_multi_temp = self.target_multiclass[X_temp.shape[0] - len(y_bin_temp):]
            y_multi_test = self.target_multiclass[X_test.shape[0] - len(y_bin_test):]
            
            y_multi_train = self.target_multiclass[:len(y_bin_train)]
            y_multi_val = self.target_multiclass[len(y_bin_train):len(y_bin_train) + len(y_bin_val)]
            
            data_splits['multiclass'] = {
                'y_train': y_multi_train,
                'y_val': y_multi_val,
                'y_test': y_multi_test
            }
        
        # Log split information
        logger.info(f"Final split sizes:")
        logger.info(f"  Train: {len(y_bin_train)} samples")
        logger.info(f"  Validation: {len(y_bin_val)} samples")
        logger.info(f"  Test: {len(y_bin_test)} samples")
        
        return data_splits
    
    def create_federated_splits(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """Create federated splits - DEPRECATED: Training split will be used for all clients"""
        logger.info("Federated splits creation skipped - training split will be used for all federated clients")
        return {}
    
    def save_processed_data(self, data_splits: Dict, federated_splits: Optional[Dict] = None) -> None:
        """Save all processed data to CSV files"""
        logger.info("Saving processed data to CSV format...")
        
        # Save main data splits as CSV files
        for split_type, split_data in data_splits.items():
            split_dir = os.path.join(self.config.output_dir, split_type)
            os.makedirs(split_dir, exist_ok=True)
            
            for key, array in split_data.items():
                # Handle different data types appropriately
                if 'X_' in key:  # Feature data
                    if len(array.shape) == 3:  # Sequence data (samples, timesteps, features)
                        # For sequence data, reshape to 2D for CSV storage
                        # Format: sample_id, timestep, feature1, feature2, ...
                        samples, timesteps, features = array.shape
                        reshaped_data = []
                        
                        for sample_idx in range(samples):
                            for timestep in range(timesteps):
                                row = [sample_idx, timestep] + array[sample_idx, timestep, :].tolist()
                                reshaped_data.append(row)
                        
                        columns = ['sample_id', 'timestep'] + [f'feature_{i}' for i in range(features)]
                        df = pd.DataFrame(reshaped_data, columns=columns)
                        file_path = os.path.join(split_dir, f"{key}_sequences.csv")
                    else:  # Tabular data (samples, features)
                        df = pd.DataFrame(array, columns=self.feature_names)
                        file_path = os.path.join(split_dir, f"{key}.csv")
                elif 'y_' in key:  # Target data
                    df = pd.DataFrame(array, columns=['target'])
                    file_path = os.path.join(split_dir, f"{key}.csv")
                else:
                    # For other arrays, save as generic CSV
                    df = pd.DataFrame(array)
                    file_path = os.path.join(split_dir, f"{key}.csv")
                
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {key} with shape {array.shape} to {file_path}")
        
        # Note: Federated splits are no longer created as training split will be used for all clients
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'num_classes_binary': len(np.unique(self.target_binary)) if self.target_binary is not None else None,
            'num_classes_multiclass': len(np.unique(self.target_multiclass)) if self.target_multiclass is not None else None,
            'config': self.config.__dict__
        }
        
        import json
        metadata_path = os.path.join(self.config.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def get_data_summary(self) -> Dict:
        """Generate comprehensive data summary"""
        summary = {
            'dataset_info': {
                'total_samples': len(self.data),
                'num_features': len(self.feature_names),
                'feature_names': self.feature_names
            },
            'target_info': {},
            'class_distributions': {}
        }
        
        if self.target_binary is not None:
            summary['target_info']['binary'] = {
                'num_classes': len(np.unique(self.target_binary)),
                'classes': np.unique(self.target_binary).tolist()
            }
            summary['class_distributions']['binary'] = np.bincount(self.target_binary).tolist()
        
        if self.target_multiclass is not None:
            summary['target_info']['multiclass'] = {
                'num_classes': len(np.unique(self.target_multiclass)),
                'classes': np.unique(self.target_multiclass).tolist()
            }
            summary['class_distributions']['multiclass'] = np.bincount(self.target_multiclass).tolist()
        
        return summary

def main():
    """Main function to run the complete data preparation pipeline"""
    # Initialize configuration
    config = DataConfig()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    try:
        # Load and clean data
        preprocessor.load_data()
        preprocessor.clean_data()
        
        # Engineer features
        preprocessor.engineer_features()
        
        # Prepare features and targets
        preprocessor.prepare_features_and_targets()
        
        # Split data
        data_splits = preprocessor.split_data()
        
        # Create federated splits (deprecated - training split used for all clients)
        federated_splits = preprocessor.create_federated_splits(
            data_splits['tabular']['X_train'], 
            data_splits['tabular']['y_train']
        )
        
        # Save processed data in CSV format
        preprocessor.save_processed_data(data_splits)
        
        # Generate and print summary
        summary = preprocessor.get_data_summary()
        logger.info("Data preparation completed successfully!")
        logger.info(f"Summary: {summary}")
        
        return data_splits, federated_splits, summary
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
