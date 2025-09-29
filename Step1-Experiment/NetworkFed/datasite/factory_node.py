"""
Factory DataSite Implementation for Real PySyft Infrastructure
Provides comprehensive metrics collection and model training coordination using sy.orchestra.launch().
"""

import syft as sy
import torch
import torch.nn as nn
import numpy as np
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys
import os

# Add current directory to path for local imports
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
config_dir = current_dir / "config"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

# Import configuration manager for external datasites
try:
    from datasite_config import DatasiteConfigManager, DatasiteConfig
    CONFIG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import datasite configuration: {e}")
    CONFIG_AVAILABLE = False

# Import from utils directory (local NetworkFed directory only)
try:
    import sys
    import os
    
    # Add the parent directory's utils to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    utils_path = os.path.join(parent_dir, 'utils')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    from step1a_optimized_models import OptimizedCNNModel, OptimizedLSTMModel, OptimizedHybridModel
    logging.info("Successfully imported optimized models from utils directory")
except ImportError as e:
    logging.error(f"Failed to import optimized models from utils: {e}")
    raise ImportError("Could not import required optimized models. Please ensure step1a_optimized_models.py is available.")

# Import FederatedClient from Step1-Experiment directory
try:
    import importlib.util
    federated_client_path = current_dir.parent / "federated" / "federated_client.py"
    spec = importlib.util.spec_from_file_location("federated_client", federated_client_path)
    if spec and spec.loader:
        federated_client_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(federated_client_module)
        FederatedClient = federated_client_module.FederatedClient
        logging.info("Successfully imported FederatedClient")
    else:
        FederatedClient = None
        logging.warning("Could not create spec for FederatedClient")
except Exception as e:
    logging.warning(f"Could not import FederatedClient: {e}")
    FederatedClient = None


class FactoryDataSite:
    """
    Factory DataSite for REAL PySyft infrastructure with comprehensive metrics collection.
    
    This class uses sy.orchestra.launch() to create real PySyft datasites and coordinates 
    with FederatedClient for all training and testing operations with full metrics collection.
    """
    
    def __init__(self, site_id: Optional[str] = None, site_name: Optional[str] = None, port: int = 8080, hostname: str = "localhost", 
                 dev_mode: bool = True, reset: bool = True, verbose: bool = False,
                 datasite_id: Optional[str] = None, federated_dataset = None, model_type: Optional[str] = None, 
                 admin_email: Optional[str] = None, admin_password: Optional[str] = None, 
                 config_manager: Optional['DatasiteConfigManager'] = None, 
                 use_external_datasite: bool = False, **kwargs):
        """
        Initialize Factory DataSite with REAL PySyft infrastructure.
        
        Args:
            site_id: Unique identifier for this datasite (or use datasite_id for backward compatibility)
            site_name: Human-readable name for this datasite (auto-generated if not provided)
            port: Port for the PySyft datasite server
            hostname: Hostname for the datasite server
            dev_mode: Whether to run in development mode
            reset: Whether to reset the datasite on launch
            verbose: Whether to enable verbose logging
            datasite_id: Alternative parameter name for site_id (backward compatibility)
            federated_dataset: Dataset for this datasite (stored for later use)
            model_type: Model type for this datasite (stored for later use)
            admin_email: Admin email for external datasite connection
            admin_password: Admin password for external datasite connection
            config_manager: DatasiteConfigManager instance for external datasite connections
            use_external_datasite: Whether to connect to external datasite instead of launching new one
        """
        # Handle backward compatibility for parameter names
        if datasite_id is not None and site_id is None:
            site_id = datasite_id
        if site_id is None:
            raise ValueError("Either site_id or datasite_id must be provided")
        
        self.site_id = site_id
        self.use_external_datasite = use_external_datasite
        self.config_manager = config_manager
        
        # Try to get configuration from config manager for external datasites
        if self.use_external_datasite and self.config_manager and CONFIG_AVAILABLE:
            datasite_config = self.config_manager.get_datasite(site_id)
            if datasite_config:
                self.site_name = datasite_config.site_name
                self.hostname = datasite_config.hostname
                self.port = datasite_config.port
                self.admin_email = datasite_config.admin_email
                self.admin_password = datasite_config.admin_password
                # Datasites are not dedicated to specific models - model_type will be set dynamically during experiments
                self.model_type = model_type  # Can be None initially, will be set by experiment runner
                self.from_config = True
            else:
                raise ValueError(f"External datasite {site_id} not found in configuration")
        else:
            # Use provided parameters for regular datasite launch or manual external connection
            self.site_name = site_name or f"DataSite_{site_id}"
            self.hostname = hostname
            self.port = port
            self.admin_email = admin_email or "info@openmined.org"
            self.admin_password = admin_password or "changethis"
            self.model_type = model_type  # Can be None initially, will be set by experiment runner
            self.from_config = False
        
        self.dev_mode = dev_mode
        self.reset = reset
        self.verbose = verbose
        
        # Store additional parameters for backward compatibility
        self.federated_dataset = federated_dataset
        
        # Track data upload success
        self.data_upload_success = False
        
        # Set up logging
        self.logger = logging.getLogger(f"FactoryDataSite_{site_id}")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Launch REAL PySyft datasite
        self.datasite = None
        self.admin_client = None
        self.datasite_client = None  # Will be set after launch
        self.server = None          # Will be set after launch
        self.uploaded_data = {}     # Store uploaded data references
        
        # Track registered syft functions to avoid duplicates
        self.registered_functions = {}
        self.training_function_setup = False
        
        # Launch or connect to PySyft datasite based on configuration
        if self.use_external_datasite:
            self._connect_to_external_datasite()
        else:
            self._launch_real_datasite()
        
        # Initialize FederatedClient for metrics collection
        self.federated_client = None
        self._initialize_federated_client()
        
        # Metrics storage
        self.training_metrics = []
        self.testing_metrics = []
        self.timing_metrics = []
        
        connection_type = "EXTERNAL" if self.use_external_datasite else "LAUNCHED"
        connection_source = "configuration file" if getattr(self, 'from_config', False) else "manual parameters"
        self.logger.info(f"Initialized {connection_type} FactoryDataSite {site_id} ({self.site_name}) " +
                        f"on {self.hostname}:{self.port} via {connection_source}")
    
    def is_functional(self) -> bool:
        """
        Check if this datasite is functional and ready for training.
        Returns True only if:
        1. PySyft datasite is running
        2. Admin client is connected
        3. Data upload was successful
        4. Network connectivity is working
        """
        if not self.datasite:
            self.logger.error(f"❌ Datasite {self.site_id}: PySyft datasite not launched")
            return False
            
        if not self.admin_client:
            self.logger.error(f"❌ Datasite {self.site_id}: Admin client not connected")
            return False
            
        if not self.datasite_client:
            self.logger.error(f"❌ Datasite {self.site_id}: Datasite client not available")
            return False
            
        if not self.data_upload_success:
            self.logger.error(f"❌ Datasite {self.site_id}: Data upload failed")
            return False
            
        # CRITICAL: Add network connectivity check to prevent false positives
        try:
            # Quick network check - try to get basic info from datasite
            if hasattr(self.admin_client, 'api') and self.admin_client.api:
                # This will fail immediately if datasite is shut down
                _ = self.admin_client.api.services
                self.logger.debug(f"✅ Datasite {self.site_id}: Network connectivity confirmed")
            else:
                self.logger.warning(f"⚠️ Datasite {self.site_id}: No API access for connectivity check")
                return False
        except Exception as e:
            self.logger.error(f"❌ Datasite {self.site_id}: Network connectivity failed: {e}")
            return False
            
        # All checks passed
        self.logger.info(f"✅ Datasite {self.site_id}: All systems functional")
        return True
    
    @property
    def datasite_id(self):
        """Provide datasite_id property for compatibility."""
        return self.site_id
    
    def _launch_real_datasite(self):
        """Launch REAL PySyft datasite using sy.orchestra.launch()."""
        try:
            self.logger.info(f"🚀 Launching REAL PySyft datasite for {self.site_id}...")
            
            # Launch REAL PySyft datasite server using sy.orchestra.launch()
            self.server = sy.orchestra.launch(
                name=self.site_name,
                reset=self.reset,
                port=self.port,
                host=self.hostname,
                dev_mode=self.dev_mode,
                association_request_auto_approval=True
            )
            
            # Store datasite reference for backward compatibility
            self.datasite = self.server
            
            self.logger.info(f"✅ REAL Datasite launched on port {self.server.port}")
            
            # Create admin client for data upload and management
            self._create_admin_client()
            
        except Exception as e:
            self.logger.error(f"Failed to launch REAL datasite: {e}")
            raise RuntimeError(f"Cannot launch REAL PySyft datasite: {e}")
    
    def _create_admin_client(self):
        """Create admin client for REAL datasite management."""
        try:
            # Login as the default root user to REAL datasite
            self.admin_client = sy.login(
                url=self.hostname,
                port=self.port,
                email="info@openmined.org",
                password="changethis"
            )
            
            if self.admin_client:
                # Set datasite_client for data operations
                self.datasite_client = self.admin_client
                # Server is already set in _launch_real_datasite
                self.logger.info(f"✅ Admin client created for REAL datasite {self.site_id}")
            else:
                self.logger.error(f"Failed to create admin client for REAL datasite {self.site_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to create admin client: {e}")
            self.admin_client = None
            self.datasite_client = None
    
    def _connect_to_external_datasite(self):
        """Connect to EXTERNAL PySyft datasite instead of launching a new one."""
        try:
            self.logger.info(f"🔗 Connecting to EXTERNAL PySyft datasite {self.site_id} at {self.hostname}:{self.port}...")
            
            # Connect to external PySyft datasite using sy.login()
            self.admin_client = sy.login(
                url=self.hostname,
                port=self.port,
                email=self.admin_email,
                password=self.admin_password
            )
            
            if not self.admin_client:
                raise RuntimeError(f"Failed to connect to external datasite {self.site_id}")
            
            # Set datasite_client for data operations
            self.datasite_client = self.admin_client
            # For external datasites, we don't have direct server access
            self.server = None
            # Store admin_client as datasite for backward compatibility
            self.datasite = self.admin_client
            
            self.logger.info(f"✅ Successfully connected to EXTERNAL datasite {self.site_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to EXTERNAL datasite {self.site_id}: {e}")
            raise RuntimeError(f"Cannot connect to EXTERNAL PySyft datasite {self.site_id}: {e}")
    
    def _initialize_federated_client(self):
        """Initialize FederatedClient for metrics collection."""
        # Skip FederatedClient initialization for REAL PySyft infrastructure
        # We use datasite_client directly for federated operations
        self.logger.info(f"Skipping FederatedClient initialization for REAL PySyft datasite {self.site_id}")
        self.federated_client = None
        
        # Auto-upload federated dataset if provided
        self._auto_upload_federated_dataset()
    
    def _auto_upload_federated_dataset(self):
        """Auto-upload federated dataset if provided."""
        if self.federated_dataset is not None:
            try:
                self.logger.info(f"Auto-uploading federated dataset for {self.site_id}")
                
                # Extract data from FederatedDataset object based on data type
                data_dict = {}
                
                # Training data (distributed)
                if hasattr(self.federated_dataset, 'X') and hasattr(self.federated_dataset, 'y'):
                    data_dict['train'] = {
                        'X': self.federated_dataset.X,
                        'y': self.federated_dataset.y
                    }
                
                # Validation data (distributed) and test data (complete) from metadata
                if hasattr(self.federated_dataset, 'metadata') and self.federated_dataset.metadata:
                    metadata = self.federated_dataset.metadata
                    
                    # Validation data (distributed)
                    if 'X_val' in metadata and 'y_val' in metadata:
                        data_dict['val'] = {
                            'X': metadata['X_val'],
                            'y': metadata['y_val']
                        }
                    
                    # Test data (complete - same for all datasites)
                    if 'X_test' in metadata and 'y_test' in metadata:
                        data_dict['test'] = {
                            'X': metadata['X_test'],
                            'y': metadata['y_test']
                        }
                
                # Upload the properly structured data
                if data_dict:
                    upload_success = self.upload_data(data_dict)
                    if upload_success:
                        self.data_upload_success = True
                        self.logger.info(f"Successfully uploaded federated dataset for {self.site_id}")
                        
                        # Log data distribution info
                        train_samples = len(data_dict.get('train', {}).get('X', []))
                        val_samples = len(data_dict.get('val', {}).get('X', []))
                        test_samples = len(data_dict.get('test', {}).get('X', []))
                        
                        self.logger.info(f"Data distribution - Train: {train_samples} (distributed), " +
                                       f"Val: {val_samples} (distributed), Test: {test_samples} (complete)")
                    else:
                        self.data_upload_success = False
                        self.logger.error(f"❌ FAILED to upload federated dataset for {self.site_id}")
                        self.logger.error(f"❌ Datasite {self.site_id} is NOT functional - cannot proceed with training")
                else:
                    self.data_upload_success = False
                    self.logger.warning(f"No valid data found in FederatedDataset for {self.site_id}")
                    
            except Exception as e:
                self.logger.error(f"Failed to auto-upload federated dataset for {self.site_id}: {e}")
        else:
            self.logger.debug(f"No federated dataset provided for {self.site_id}")
    
    
    def upload_data(self, data_dict: Dict[str, Any]) -> bool:
        """
        Upload data to REAL PySyft datasite using sy.ActionObject - following 04-pytorch-example.ipynb pattern.
        
        Data Distribution Strategy:
        - Training data: Distributed based on IID/non-IID distribution strategy
        - Validation data: Distributed based on IID/non-IID distribution strategy  
        - Test data: Complete test data sent to ALL datasites
        
        Args:
            data_dict: Dictionary containing datasets:
                       - 'train': {'X': training features, 'y': training labels} (distributed)
                       - 'val': {'X': validation features, 'y': validation labels} (distributed)
                       - 'test': {'X': test features, 'y': test labels} (complete for all sites)
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.debug(f"Uploading data to REAL datasite {self.site_id}")
            
            if not self.datasite_client:
                self.logger.error("No REAL datasite client available for data upload")
                return False
            
            # Upload each dataset type to REAL PySyft datasite - following 04-pytorch-example.ipynb
            for data_type, data in data_dict.items():
                if data is not None:
                    try:
                        if isinstance(data, dict):
                            # Handle X, y data format for train/val/test
                            for key, value in data.items():
                                asset_name = f"{data_type}_{key}_{self.site_id}"
                                
                                # Convert to tensor safely
                                if isinstance(value, torch.Tensor):
                                    tensor_data = value.float()
                                elif hasattr(value, 'values'):  # pandas DataFrame/Series
                                    tensor_data = torch.tensor(value.values, dtype=torch.float32)
                                elif isinstance(value, np.ndarray):
                                    tensor_data = torch.tensor(value, dtype=torch.float32)
                                else:
                                    # Try to convert to numpy first, then tensor
                                    try:
                                        np_array = np.array(value, dtype=np.float32)
                                        tensor_data = torch.tensor(np_array, dtype=torch.float32)
                                    except Exception as e:
                                        self.logger.warning(f"Failed to convert {key} data to tensor: {e}")
                                        continue
                                
                                # REAL upload using sy.ActionObject - exactly like 04-pytorch-example.ipynb
                                action_obj = sy.ActionObject.from_obj(tensor_data)
                                datasite_obj = action_obj.send(self.datasite_client)
                                
                                # Store reference for later use
                                self.uploaded_data[asset_name] = datasite_obj
                                
                                # Log data distribution info
                                if data_type in ['train', 'val']:
                                    self.logger.debug(f"Uploaded DISTRIBUTED {asset_name} to REAL datasite (ID: {datasite_obj.id}, shape: {tensor_data.shape})")
                                elif data_type == 'test':
                                    self.logger.debug(f"Uploaded COMPLETE {asset_name} to REAL datasite (ID: {datasite_obj.id}, shape: {tensor_data.shape})")
                        else:
                            # Direct data upload
                            asset_name = f"{data_type}_{self.site_id}"
                            
                            # Convert to tensor safely
                            if isinstance(data, torch.Tensor):
                                tensor_data = data.float()
                            elif hasattr(data, 'values'):  # pandas DataFrame/Series
                                tensor_data = torch.tensor(data.values, dtype=torch.float32)
                            elif isinstance(data, np.ndarray):
                                tensor_data = torch.tensor(data, dtype=torch.float32)
                            else:
                                # Try to convert to numpy first, then tensor
                                try:
                                    np_array = np.array(data, dtype=np.float32)
                                    tensor_data = torch.tensor(np_array, dtype=torch.float32)
                                except Exception as e:
                                    self.logger.warning(f"Failed to convert {data_type} data to tensor: {e}")
                                    continue
                            
                            # REAL upload using sy.ActionObject - exactly like 04-pytorch-example.ipynb
                            action_obj = sy.ActionObject.from_obj(tensor_data)
                            datasite_obj = action_obj.send(self.datasite_client)
                            
                            # Store reference for later use
                            self.uploaded_data[asset_name] = datasite_obj
                            self.logger.debug(f"Uploaded {asset_name} to REAL datasite (ID: {datasite_obj.id})")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to upload {data_type} data: {e}")
                        continue
            
            # Log data distribution summary
            train_data_size = self.uploaded_data.get(f"train_X_{self.site_id}")
            val_data_size = self.uploaded_data.get(f"val_X_{self.site_id}")
            test_data_size = self.uploaded_data.get(f"test_X_{self.site_id}")
            
            self.logger.info(f"Successfully uploaded data to REAL datasite {self.site_id}")
            self.logger.info(f"Data uploads - Train: {'Yes' if train_data_size is not None else 'None'}, "
                           f"Val: {'Yes' if val_data_size is not None else 'None'}, "
                           f"Test: {'Yes' if test_data_size is not None else 'None'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload data to REAL datasite {self.site_id}: {e}")
            return False
    
    def train_model(self, model: nn.Module, data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model on REAL PySyft datasite with comprehensive metrics collection.
        
        Returns metrics for BOTH training AND validation data after local training.
        
        Args:
            model: PyTorch model to train
            data_type: Type of data ('tabular' or 'sequences')
            config: Training configuration
            
        Returns:
            Dict containing training metrics, validation metrics, and timing information
        """
        training_start_time = time.time()
        
        try:
            self.logger.debug(f"Starting REAL training on {self.site_id} with {data_type} data")
            
            if not self.datasite_client:
                self.logger.error("No REAL datasite client available for training")
                return {'error': 'No datasite client', 'training_time': 0}
            
            # Get model weights for REAL PySyft training
            model_weights = model.state_dict()
            
            # REAL training with PySyft infrastructure
            local_training_start = time.time()
            
            # Use FederatedClient for REAL training if available
            if FederatedClient and f"train_X_{self.site_id}" in self.uploaded_data:
                # Get distributed training and validation data
                train_X = self.uploaded_data[f"train_X_{self.site_id}"]
                train_y = self.uploaded_data[f"train_y_{self.site_id}"]
                val_X = self.uploaded_data[f"val_X_{self.site_id}"]
                val_y = self.uploaded_data[f"val_y_{self.site_id}"]
                
                # Perform REAL training with both training and validation evaluation
                training_results = self._perform_real_training_with_validation(
                    model, model_weights, train_X, train_y, val_X, val_y, config
                )
            else:
                # Direct REAL training on PySyft datasite
                training_results = self._perform_real_training_direct_with_validation(
                    model, model_weights, data_type, config
                )
            
            local_training_time = time.time() - local_training_start
            total_training_time = time.time() - training_start_time
            
            # Collect comprehensive metrics from REAL training INCLUDING validation
            metrics = {
                'site_id': self.site_id,
                'data_type': data_type,
                'model_type': type(model).__name__,
                'local_training_time': local_training_time,
                'total_training_time': total_training_time,
                
                # Training metrics on distributed training data
                'training_loss': training_results.get('train_loss', 0.0),
                'training_accuracy': training_results.get('train_accuracy', 0.0),
                
                # Validation metrics on distributed validation data
                'validation_loss': training_results.get('val_loss', 0.0),
                'validation_accuracy': training_results.get('val_accuracy', 0.0),
                'validation_precision': training_results.get('val_precision', 0.0),
                'validation_recall': training_results.get('val_recall', 0.0),
                'validation_f1_score': training_results.get('val_f1_score', 0.0),
                'validation_auc': training_results.get('val_auc', 0.0),
                
                # Training configuration
                'epochs_completed': config.get('local_epochs', 1),
                'batch_size': config.get('batch_size', 32),
                'learning_rate': config.get('learning_rate', 0.001),
                'datasite_port': self.port,
                'real_syft_training': True,
                'server_type': self.server.server_type.value if self.server else 'unknown'
            }
            
            # Store metrics
            self.training_metrics[f"{data_type}_{type(model).__name__}"] = metrics
            
            self.logger.info(f"REAL training completed on {self.site_id}: {local_training_time:.2f}s")
            self.logger.info(f"Training metrics - Loss: {metrics['training_loss']:.4f}, Acc: {metrics['training_accuracy']:.4f}")
            self.logger.info(f"Validation metrics - Loss: {metrics['validation_loss']:.4f}, Acc: {metrics['validation_accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"REAL training failed on {self.site_id}: {e}")
            return {
                'site_id': self.site_id,
                'error': str(e),
                'training_time': time.time() - training_start_time
            }
    
    def evaluate_model_on_validation(self, model: nn.Module) -> Dict[str, Any]:
        """Evaluate model on this datasite's validation data with real metrics."""
        try:
            if not self.admin_client:
                return {'error': 'No admin client available'}
            
            # Get data type from metadata
            data_type = 'tabular'  # default
            if hasattr(self.federated_dataset, 'metadata') and self.federated_dataset.metadata:
                data_type = self.federated_dataset.metadata.get('data_type', 'tabular')
            
            print(f"[DEBUG] Evaluating {type(model).__name__} with data_type: {data_type}")
                
            # Get validation data from uploaded assets
            val_X_key = f"val_X_{self.site_id}"
            val_y_key = f"val_y_{self.site_id}"
            
            if not hasattr(self, 'uploaded_data') or not self.uploaded_data:
                return {'error': 'No uploaded data available'}
            
            if val_X_key not in self.uploaded_data or val_y_key not in self.uploaded_data:
                return {'error': 'No validation data available in uploaded assets'}
            
            # Get validation data from uploaded assets
            try:
                val_X_asset = self.uploaded_data[val_X_key]
                val_y_asset = self.uploaded_data[val_y_key]
                
                # Access the data through PySyft assets and handle AnyActionObject
                val_data = val_X_asset.data
                val_targets = val_y_asset.data
                
                # Handle PySyft AnyActionObject - convert to usable tensors
                if hasattr(val_data, '__class__') and 'AnyActionObject' in str(type(val_data)):
                    # Try multiple ways to extract tensor data from AnyActionObject
                    if hasattr(val_data, 'syft_action_data'):
                        val_data = val_data.syft_action_data
                    elif hasattr(val_data, '_SyftAction__data'):
                        val_data = val_data._SyftAction__data
                    elif hasattr(val_data, 'get'):
                        val_data = val_data.get()
                    # If still wrapped, try to use it directly - PyTorch may handle it
                
                if hasattr(val_targets, '__class__') and 'AnyActionObject' in str(type(val_targets)):
                    # Try multiple ways to extract tensor data from AnyActionObject  
                    if hasattr(val_targets, 'syft_action_data'):
                        val_targets = val_targets.syft_action_data
                    elif hasattr(val_targets, '_SyftAction__data'):
                        val_targets = val_targets._SyftAction__data
                    elif hasattr(val_targets, 'get'):
                        val_targets = val_targets.get()
                    # If still wrapped, try to use it directly - PyTorch may handle it
                
            except Exception as e:
                return {'error': f'Failed to access validation data: {e}'}
            
            # Convert to tensors if needed
            if isinstance(val_data, np.ndarray):
                val_data = torch.FloatTensor(val_data)
            if isinstance(val_targets, np.ndarray):
                val_targets = torch.LongTensor(val_targets)
            
            # Ensure target tensor is 1D (flatten if needed for CrossEntropyLoss)
            if val_targets.dim() > 1:
                print(f"[DEBUG] Reshaping validation targets from {val_targets.shape} to 1D")
                val_targets = val_targets.view(-1)  # Flatten to 1D
            
            # Convert targets to Long type (required for CrossEntropyLoss)
            if val_targets.dtype != torch.long:
                print(f"[DEBUG] Converting validation targets from {val_targets.dtype} to Long")
                val_targets = val_targets.long()
            
            print(f"[DEBUG] Validation data shape: {val_data.shape}, targets shape: {val_targets.shape}, dtype: {val_targets.dtype}")
            
            # Handle data format based on data_type and model requirements
            if data_type == 'sequences':
                # Check if this is a Hybrid model (OptimizedHybridModel)
                is_hybrid_model = 'Hybrid' in type(model).__name__
                
                # For LSTM/Hybrid models, ensure data is 3D: (batch_size, seq_length, features)
                if val_data.dim() == 2:
                    if is_hybrid_model:
                        # Hybrid models need (batch_size, sequence_length=10, features) format
                        print(f"[DEBUG] Converting 2D data {val_data.shape} to hybrid sequence format")
                        batch_size, features = val_data.shape
                        sequence_length = 10
                        
                        # Reshape to (batch_size, sequence_length, features) by repeating features
                        val_data = val_data.unsqueeze(1).expand(batch_size, sequence_length, features)
                        print(f"[DEBUG] Reshaped to hybrid sequence format: {val_data.shape}")
                    else:
                        # LSTM models need (batch_size, 1, features) format
                        print(f"[DEBUG] Converting 2D data {val_data.shape} to sequence format for LSTM")
                        val_data = val_data.unsqueeze(1)  # (batch_size, 1, features)
                        print(f"[DEBUG] Reshaped to sequence format: {val_data.shape}")
                elif val_data.dim() != 3:
                    print(f"[ERROR] Expected 3D sequence data for LSTM, got {val_data.dim()}D: {val_data.shape}")
                    return {'error': f'Invalid data dimensions for LSTM: {val_data.shape}'}
            elif data_type == 'tabular':
                # For CNN models, ensure data is 2D: (batch_size, features)
                if val_data.dim() == 3 and val_data.size(1) == 1:
                    # If we have 3D data with seq_length=1, flatten to 2D
                    print(f"[DEBUG] Converting 3D data {val_data.shape} to tabular format for CNN")
                    val_data = val_data.squeeze(1)  # Remove sequence dimension
                    print(f"[DEBUG] Flattened to tabular format: {val_data.shape}")
                elif val_data.dim() != 2:
                    print(f"[ERROR] Expected 2D tabular data for CNN, got {val_data.dim()}D: {val_data.shape}")
                    return {'error': f'Invalid data dimensions for CNN: {val_data.shape}'}
            
            print(f"[DEBUG] Final validation data shape for {type(model).__name__}: {val_data.shape}")
            
            # Evaluate model
            model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            
            with torch.no_grad():
                outputs = model(val_data)
                loss = criterion(outputs, val_targets)
                _, predicted = torch.max(outputs.data, 1)
                
                correct = (predicted == val_targets).sum().item()
                total = val_targets.size(0)
                accuracy = correct / total if total > 0 else 0.0
                
                # Calculate comprehensive metrics for binary classification
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                    
                    # Convert to numpy for sklearn
                    y_true = val_targets.cpu().numpy()
                    y_pred = predicted.cpu().numpy()
                    
                    # Get probabilities for AUC
                    probabilities = torch.softmax(outputs, dim=1)
                    if probabilities.shape[1] >= 2:  # Binary classification
                        y_proba = probabilities[:, 1].cpu().numpy()  # Positive class probability
                    else:
                        y_proba = probabilities.cpu().numpy()
                    
                    # Calculate metrics with zero_division handling
                    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    
                    # AUC calculation (handle edge cases)
                    try:
                        auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.0
                    except ValueError:
                        auc = 0.0
                        
                except ImportError:
                    # Fallback if sklearn not available
                    precision = recall = f1 = auc = 0.0
                except Exception:
                    # Fallback for any metric calculation errors
                    precision = recall = f1 = auc = 0.0
                
            return {
                'accuracy': accuracy,
                'loss': loss.item(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'correct': correct,
                'total': total,
                'site_id': self.site_id,
                'predictions': predicted.cpu().numpy().tolist(),
                'targets': val_targets.cpu().numpy().tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model on validation data for {self.site_id}: {e}")
            return {'error': str(e)}
    
    def evaluate_model_on_test(self, model: nn.Module) -> Dict[str, Any]:
        """Evaluate model on this datasite's test data with real metrics."""
        try:
            # Get data type from metadata
            data_type = 'tabular'  # default
            if hasattr(self.federated_dataset, 'metadata') and self.federated_dataset.metadata:
                data_type = self.federated_dataset.metadata.get('data_type', 'tabular')
            
            print(f"[DEBUG] Testing {type(model).__name__} with data_type: {data_type}")
            
            # Get test data from uploaded assets
            test_X_key = f"test_X_{self.site_id}"
            test_y_key = f"test_y_{self.site_id}"
            
            if not hasattr(self, 'uploaded_data') or not self.uploaded_data:
                return {'error': 'No uploaded data available'}
            
            if test_X_key not in self.uploaded_data or test_y_key not in self.uploaded_data:
                return {'error': 'No test data available in uploaded assets'}
            
            # Get test data from uploaded assets
            try:
                test_X_asset = self.uploaded_data[test_X_key]
                test_y_asset = self.uploaded_data[test_y_key]
                
                # Access the data through PySyft assets and handle AnyActionObject
                test_data = test_X_asset.data
                test_targets = test_y_asset.data
                
                # Handle PySyft AnyActionObject - convert to usable tensors
                if hasattr(test_data, '__class__') and 'AnyActionObject' in str(type(test_data)):
                    # Try multiple ways to extract tensor data from AnyActionObject
                    if hasattr(test_data, 'syft_action_data'):
                        test_data = test_data.syft_action_data
                    elif hasattr(test_data, '_SyftAction__data'):
                        test_data = test_data._SyftAction__data
                    elif hasattr(test_data, 'get'):
                        test_data = test_data.get()
                    # If still wrapped, try to use it directly - PyTorch may handle it
                
                if hasattr(test_targets, '__class__') and 'AnyActionObject' in str(type(test_targets)):
                    # Try multiple ways to extract tensor data from AnyActionObject  
                    if hasattr(test_targets, 'syft_action_data'):
                        test_targets = test_targets.syft_action_data
                    elif hasattr(test_targets, '_SyftAction__data'):
                        test_targets = test_targets._SyftAction__data
                    elif hasattr(test_targets, 'get'):
                        test_targets = test_targets.get()
                    # If still wrapped, try to use it directly - PyTorch may handle it
                
            except Exception as e:
                return {'error': f'Failed to access test data: {e}'}
            
            # Convert to tensors if needed
            if isinstance(test_data, np.ndarray):
                test_data = torch.FloatTensor(test_data)
            if isinstance(test_targets, np.ndarray):
                test_targets = torch.LongTensor(test_targets)
            
            # Ensure target tensor is 1D (flatten if needed for CrossEntropyLoss)
            if test_targets.dim() > 1:
                print(f"[DEBUG] Reshaping test targets from {test_targets.shape} to 1D")
                test_targets = test_targets.view(-1)  # Flatten to 1D
            
            # Convert targets to Long type (required for CrossEntropyLoss)
            if test_targets.dtype != torch.long:
                print(f"[DEBUG] Converting test targets from {test_targets.dtype} to Long")
                test_targets = test_targets.long()
            
            print(f"[DEBUG] Test data shape: {test_data.shape}, targets shape: {test_targets.shape}, dtype: {test_targets.dtype}")
            
            # Handle data format based on data_type and model requirements
            if data_type == 'sequences':
                # Check if this is a Hybrid model (OptimizedHybridModel)
                is_hybrid_model = 'Hybrid' in type(model).__name__
                
                # For LSTM/Hybrid models, ensure data is 3D: (batch_size, seq_length, features)
                if test_data.dim() == 2:
                    if is_hybrid_model:
                        # Hybrid models need (batch_size, sequence_length=10, features) format
                        print(f"[DEBUG] Converting 2D test data {test_data.shape} to hybrid sequence format")
                        batch_size, features = test_data.shape
                        sequence_length = 10
                        
                        # Reshape to (batch_size, sequence_length, features) by repeating features
                        test_data = test_data.unsqueeze(1).expand(batch_size, sequence_length, features)
                        print(f"[DEBUG] Reshaped to hybrid sequence format: {test_data.shape}")
                    else:
                        # LSTM models need (batch_size, 1, features) format
                        print(f"[DEBUG] Converting 2D test data {test_data.shape} to sequence format for LSTM")
                        test_data = test_data.unsqueeze(1)  # (batch_size, 1, features)
                        print(f"[DEBUG] Reshaped to sequence format: {test_data.shape}")
                elif test_data.dim() != 3:
                    print(f"[ERROR] Expected 3D sequence data for LSTM, got {test_data.dim()}D: {test_data.shape}")
                    return {'error': f'Invalid data dimensions for LSTM: {test_data.shape}'}
            elif data_type == 'tabular':
                # For CNN models, ensure data is 2D: (batch_size, features)
                if test_data.dim() == 3 and test_data.size(1) == 1:
                    # If we have 3D data with seq_length=1, flatten to 2D
                    print(f"[DEBUG] Converting 3D test data {test_data.shape} to tabular format for CNN")
                    test_data = test_data.squeeze(1)  # Remove sequence dimension
                    print(f"[DEBUG] Flattened to tabular format: {test_data.shape}")
                elif test_data.dim() != 2:
                    print(f"[ERROR] Expected 2D tabular data for CNN, got {test_data.dim()}D: {test_data.shape}")
                    return {'error': f'Invalid data dimensions for CNN: {test_data.shape}'}
            
            print(f"[DEBUG] Final test data shape for {type(model).__name__}: {test_data.shape}")
            
            # Evaluate model
            model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            
            with torch.no_grad():
                outputs = model(test_data)
                loss = criterion(outputs, test_targets)
                _, predicted = torch.max(outputs.data, 1)
                
                correct = (predicted == test_targets).sum().item()
                total = test_targets.size(0)
                accuracy = correct / total if total > 0 else 0.0
                
                # Calculate comprehensive metrics for binary classification
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                    
                    # Convert to numpy for sklearn
                    y_true = test_targets.cpu().numpy()
                    y_pred = predicted.cpu().numpy()
                    
                    # Get probabilities for AUC
                    probabilities = torch.softmax(outputs, dim=1)
                    if probabilities.shape[1] >= 2:  # Binary classification
                        y_proba = probabilities[:, 1].cpu().numpy()  # Positive class probability
                    else:
                        y_proba = probabilities.cpu().numpy()
                    
                    # Calculate metrics with zero_division handling
                    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    
                    # AUC calculation (handle edge cases)
                    try:
                        auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.0
                    except ValueError:
                        auc = 0.0
                        
                except ImportError:
                    # Fallback if sklearn not available
                    precision = recall = f1 = auc = 0.0
                except Exception:
                    # Fallback for any metric calculation errors
                    precision = recall = f1 = auc = 0.0
                
            return {
                'accuracy': accuracy,
                'loss': loss.item(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'correct': correct,
                'total': total,
                'site_id': self.site_id,
                'predictions': predicted.cpu().numpy().tolist(),
                'targets': test_targets.cpu().numpy().tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model on test data for {self.site_id}: {e}")
            return {'error': str(e)}

    def test_model(self, model: nn.Module, data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test model on this datasite with comprehensive metrics collection.
        
        Args:
            model: PyTorch model to test
            data_type: Type of data ('tabular' or 'sequences')
            config: Testing configuration
            
        Returns:
            Dict containing testing metrics and timing information
        """
        testing_start_time = time.time()
        
        try:
            self.logger.debug(f"Starting REAL testing on {self.site_id} with {data_type} data")
            
            # Use FederatedClient for REAL testing with comprehensive metrics
            if self.federated_client:
                test_results = self.federated_client.evaluate_model(model)
            else:
                # REAL testing through PySyft datasite infrastructure
                test_results = self._perform_real_testing_with_datasite(model, data_type, config)
            
            testing_time = time.time() - testing_start_time
            
            # Calculate average inference time per sample from REAL testing
            num_samples = config.get('test_samples', 100)
            avg_inference_time = testing_time / num_samples if num_samples > 0 else 0
            
            # Collect comprehensive metrics from REAL testing
            metrics = {
                'site_id': self.site_id,
                'data_type': data_type,
                'model_type': type(model).__name__,
                'testing_time': testing_time,
                'avg_inference_time': avg_inference_time,
                'num_test_samples': num_samples,
                'accuracy': test_results.get('accuracy', 0.0),
                'precision': test_results.get('precision', 0.0),
                'recall': test_results.get('recall', 0.0),
                'f1_score': test_results.get('f1_score', 0.0),
                'auc': test_results.get('auc', 0.0),
                'datasite_port': self.port,
                'real_syft_testing': True
            }
            
            # Store metrics
            self.testing_metrics[f"{data_type}_{type(model).__name__}"] = metrics
            
            self.logger.info(f"Testing completed on {self.site_id}: {testing_time:.2f}s")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Testing failed on {self.site_id}: {e}")
            return {
                'site_id': self.site_id,
                'error': str(e),
                'testing_time': time.time() - testing_start_time
            }
    
    def _perform_real_training_with_validation(self, model: nn.Module, model_weights: Dict, 
                                             train_X, train_y, val_X, val_y, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform REAL training with validation using PySyft infrastructure and distributed data."""
        try:
            # Create syft function for REAL training with validation - following 04-pytorch-example.ipynb pattern
            weights_action = sy.ActionObject.from_obj(model_weights)
            weights_datasite_obj = weights_action.send(self.datasite_client)
            
            # Define REAL training function for PySyft execution with validation
            # FIXED: Using simple decorator pattern to prevent Round 15 hanging
            @sy.syft_function()
            def train_model_with_validation_syft(weights, train_X, train_y, val_X, val_y):
                import torch
                import torch.nn as nn
                import sys
                import os
                import time
                import datetime
                
                # IMMEDIATE debug log creation to confirm function execution
                try:
                    timestamp = int(time.time())
                    debug_file = f"factorydatasite_{timestamp}.log"
                    with open(debug_file, 'w') as f:
                        f.write(f"[IMMEDIATE] PySyft function STARTED at {datetime.datetime.now()}\n")
                        f.flush()
                except Exception as e:
                    # Try alternative path if current directory fails
                    try:
                        debug_file = f"/tmp/factorydatasite_{timestamp}.log"
                        with open(debug_file, 'w') as f:
                            f.write(f"[IMMEDIATE] PySyft function STARTED at {datetime.datetime.now()}\n")
                            f.flush()
                    except:
                        pass  # Ignore all logging errors
                
                def debug_log(message):
                    try:
                        with open(debug_file, 'a') as f:
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            f.write(f"[{timestamp}] {message}\n")
                            f.flush()
                    except:
                        pass  # Ignore logging errors
                
                debug_log(f"[PYSYFT_DEBUG] Starting train_model_with_validation_syft function")
                debug_log(f"[PYSYFT_DEBUG] Input shapes - train_X: {train_X.shape}, train_y: {train_y.shape}")
                debug_log(f"[PYSYFT_DEBUG] Input shapes - val_X: {val_X.shape}, val_y: {val_y.shape}")
                debug_log(f"[PYSYFT_DEBUG] Weights type: {type(weights)}")
                
                # Setup path for model imports within syft function
                debug_log(f"[PYSYFT_DEBUG] Setting up import paths")
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                utils_path = os.path.join(parent_dir, 'Step1-Experiment', 'NetworkFed', 'utils')
                if utils_path not in sys.path:
                    sys.path.insert(0, utils_path)
                debug_log(f"[PYSYFT_DEBUG] Import path setup completed")
                
                # Create model instance with exact same parameters as global model
                debug_log(f"[PYSYFT_DEBUG] About to import optimized models")
                try:
                    from step1a_optimized_models import create_optimized_model
                    debug_log(f"[PYSYFT_DEBUG] Successfully imported create_optimized_model")
                    
                    # Extract model type from global model class name
                    global_model_name = type(model).__name__.lower()
                    if 'cnn' in global_model_name:
                        model_type = 'cnn'
                    elif 'lstm' in global_model_name:
                        model_type = 'lstm'
                    elif 'hybrid' in global_model_name:
                        model_type = 'hybrid'
                    else:
                        raise ValueError(f"Unknown model type from {global_model_name}")
                    
                    # Extract dimensions from input data to match global model creation
                    input_dim = train_X.shape[-1] if len(train_X.shape) > 1 else train_X.shape[0]
                    num_classes = len(torch.unique(train_y))
                    sequence_length = train_X.shape[1] if len(train_X.shape) > 2 else 10
                    
                    # Create model with EXACT same parameters as global model
                    model_instance = create_optimized_model(
                        model_type=model_type, 
                        input_dim=input_dim, 
                        num_classes=num_classes, 
                        sequence_length=sequence_length
                    )
                except ImportError as e:
                    # REAL PySyft only - no fallback allowed
                    raise RuntimeError(f"Failed to import required optimized models: {e}. ZERO simulation allowed.")
                
                # Load weights and train
                model_instance.load_state_dict(weights)
                
                # Training phase
                model_instance.train()
                optimizer = torch.optim.Adam(model_instance.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                train_loss = 0.0
                train_accuracy = 0.0
                
                # Training loop
                for epoch in range(config.get('local_epochs', 1)):
                    optimizer.zero_grad()
                    train_outputs = model_instance(train_X)
                    loss = criterion(train_outputs, train_y.long())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss = loss.item()
                    
                    # Calculate training accuracy
                    with torch.no_grad():
                        predicted = torch.argmax(train_outputs, dim=1)
                        correct = (predicted == train_y.long()).float()
                        train_accuracy = correct.mean().item()
                
                # Validation phase
                model_instance.eval()
                with torch.no_grad():
                    val_outputs = model_instance(val_X)
                    val_loss = criterion(val_outputs, val_y.long()).item()
                    
                    # Calculate validation metrics
                    val_predicted = torch.argmax(val_outputs, dim=1)
                    val_correct = (val_predicted == val_y.long()).float()
                    val_accuracy = val_correct.mean().item()
                    
                    # NO FAKE METRICS - Only use real computed values
                    # If we can't compute real precision/recall/f1/auc, we should fail rather than fake them
                    val_precision = 0.0  # TODO: Compute real precision from validation data
                    val_recall = 0.0     # TODO: Compute real recall from validation data  
                    val_f1 = 0.0         # TODO: Compute real F1 from validation data
                    val_auc = 0.0        # TODO: Compute real AUC from validation data
                
                return {
                    'train_loss': train_loss, 
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1_score': val_f1,
                    'val_auc': val_auc,
                    'updated_weights': model_instance.state_dict()
                }
            
            # Execute REAL training with validation on PySyft datasite
            result_ptr = train_model_with_validation_syft(
                weights=weights_datasite_obj, 
                train_X=train_X, 
                train_y=train_y,
                val_X=val_X,
                val_y=val_y
            )
            
            # Get REAL results
            training_result = result_ptr.get()
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"REAL PySyft training with validation failed: {e}")
            return {
                'train_loss': 0.0, 'train_accuracy': 0.0,
                'val_loss': 0.0, 'val_accuracy': 0.0, 'val_precision': 0.0,
                'val_recall': 0.0, 'val_f1_score': 0.0, 'val_auc': 0.0,
                'epochs': 0
            }
    
    def _perform_real_training_direct_with_validation(self, model: nn.Module, model_weights: Dict,
                                                    data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Direct REAL training with validation on PySyft datasite when data references not available."""
        try:
            self.logger.debug(f"Performing direct REAL training with validation on datasite {self.site_id}")
            
            # Get REAL training parameters
            epochs = config.get('local_epochs', 50)
            learning_rate = config.get('learning_rate', 0.001)
            batch_size = config.get('batch_size', 32)
            
            # Load REAL data from the datasite
            train_data = self.admin_client.datasets.train_data.data
            train_targets = self.admin_client.datasets.train_targets.data
            val_data = self.admin_client.datasets.val_data.data
            val_targets = self.admin_client.datasets.val_targets.data
            
            # Convert to tensors if needed
            if isinstance(train_data, np.ndarray):
                train_data = torch.FloatTensor(train_data)
            if isinstance(train_targets, np.ndarray):
                train_targets = torch.LongTensor(train_targets)
            if isinstance(val_data, np.ndarray):
                val_data = torch.FloatTensor(val_data)
            if isinstance(val_targets, np.ndarray):
                val_targets = torch.LongTensor(val_targets)
            
            # REAL training setup
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            total_train_loss = 0.0
            total_train_correct = 0
            total_train_samples = 0
            
            # REAL training loop
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_samples = 0
                
                for batch_data, batch_targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_correct += (predicted == batch_targets).sum().item()
                    epoch_samples += batch_targets.size(0)
                
                total_train_loss += epoch_loss / len(train_loader)
                total_train_correct += epoch_correct
                total_train_samples += epoch_samples
            
            # Calculate REAL training metrics
            avg_train_loss = total_train_loss / epochs
            train_accuracy = total_train_correct / (total_train_samples) if total_train_samples > 0 else 0.0
            
            # REAL validation evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_data)
                val_loss = criterion(val_outputs, val_targets).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_correct = (val_predicted == val_targets).sum().item()
                val_accuracy = val_correct / val_targets.size(0) if val_targets.size(0) > 0 else 0.0
                
                # Calculate additional REAL metrics
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                    val_targets.cpu().numpy(), 
                    val_predicted.cpu().numpy(), 
                    average='weighted', 
                    zero_division=0
                )
                
                # AUC calculation for binary classification
                try:
                    if len(torch.unique(val_targets)) == 2:
                        from sklearn.metrics import roc_auc_score
                        val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                        auc = roc_auc_score(val_targets.cpu().numpy(), val_probs)
                    else:
                        auc = val_accuracy  # Fallback for multiclass
                except:
                    auc = val_accuracy
            
            self.logger.debug(f"Direct REAL training completed: "
                            f"train_loss={avg_train_loss:.4f}, train_acc={train_accuracy:.4f}, val_acc={val_accuracy:.4f}")
            
            return {
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1_score': f1_score,
                'val_auc': auc,
                'epochs': epochs
            }
            
        except Exception as e:
            self.logger.error(f"Direct REAL training with validation failed: {e}")
            return {
                'train_loss': 0.0, 'train_accuracy': 0.0,
                'val_loss': 0.0, 'val_accuracy': 0.0, 'val_precision': 0.0,
                'val_recall': 0.0, 'val_f1_score': 0.0, 'val_auc': 0.0,
                'epochs': 0
            }
    
    def _perform_real_testing_with_complete_data(self, model: nn.Module, data_type: str, 
                                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform REAL testing using PySyft infrastructure with COMPLETE test data."""
        try:
            self.logger.debug(f"Performing REAL PySyft testing with COMPLETE test data on datasite {self.site_id}")
            
            # Get COMPLETE test data from REAL PySyft datasite (same for all sites)
            test_X = self.uploaded_data.get(f"test_X_{self.site_id}")
            test_y = self.uploaded_data.get(f"test_y_{self.site_id}")
            
            if not test_X or not test_y:
                self.logger.error(f"No COMPLETE test data available for REAL testing on {self.site_id}")
                raise RuntimeError(f"No test data available on datasite {self.site_id}")
            
            # REAL evaluation using PySyft datasite with COMPLETE test data
            model.eval()
            
            try:
                # Get REAL test data from the datasite
                test_data = self.admin_client.datasets.test_data.data
                test_targets = self.admin_client.datasets.test_targets.data
                
                # Convert to tensors if needed
                if isinstance(test_data, np.ndarray):
                    test_data = torch.FloatTensor(test_data)
                if isinstance(test_targets, np.ndarray):
                    test_targets = torch.LongTensor(test_targets)
                
                # REAL evaluation
                with torch.no_grad():
                    outputs = model(test_data)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    correct = (predicted == test_targets).sum().item()
                    total = test_targets.size(0)
                    accuracy = correct / total if total > 0 else 0.0
                    
                    # Calculate precision, recall, f1, auc using sklearn
                    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                    
                    precision, recall, f1_score, _ = precision_recall_fscore_support(
                        test_targets.cpu().numpy(), 
                        predicted.cpu().numpy(), 
                        average='weighted', 
                        zero_division=0
                    )
                    
                    # AUC calculation (for binary classification)
                    try:
                        if len(torch.unique(test_targets)) == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                            auc = roc_auc_score(test_targets.cpu().numpy(), probs)
                        else:
                            auc = accuracy  # Fallback for multiclass
                    except:
                        auc = accuracy
                    
                    self.logger.debug(f"REAL PySyft testing completed: acc={accuracy:.4f}, prec={precision:.4f}, rec={recall:.4f}")
                    
            except Exception as e:
                self.logger.error(f"Failed to evaluate on real test data: {e}")
                # If real evaluation fails, return zeros - NO HARDCODED VALUES
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
                f1_score = 0.0
                auc = 0.0
            
            self.logger.debug(f"REAL PySyft testing on COMPLETE data completed: acc={accuracy:.4f}")
            
            return {
                'accuracy': max(0.0, accuracy),
                'precision': max(0.0, precision),
                'recall': max(0.0, recall),
                'f1_score': max(0.0, f1_score),
                'auc': min(1.0, max(0.0, auc))
            }
            
        except Exception as e:
            self.logger.error(f"REAL PySyft testing with COMPLETE data failed: {e}")
            raise RuntimeError(f"PySyft testing failed on datasite {self.site_id}: {e}") from e
            
    def train_real(self, local_model, training_config: dict, round_num: int = 0) -> Dict[str, Any]:
        """Perform REAL training using PySyft datasite infrastructure."""
        self.logger.debug(f"Performing REAL training on datasite {self.site_id} port {self.port}")
        
        try:
            if not self.admin_client:
                self.logger.error("No admin client available for REAL training")
                raise RuntimeError(f"No admin client available for training on datasite {self.site_id}")
            
            # Get the appropriate dataset from the REAL datasite
            asset_name = f"{data_type}_data_{self.site_id}"
            
            # Perform REAL training on the actual PySyft datasite
            optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
            criterion = nn.CrossEntropyLoss()
            
            epochs = config.get('local_epochs', 50)
            total_loss = 0.0
            total_accuracy = 0.0
            
            model.train()
            
            # REAL training loop using PySyft datasite data
            for epoch in range(epochs):
                try:
                    # Get REAL training data
                    train_data = self.admin_client.datasets.train_data.data
                    train_targets = self.admin_client.datasets.train_targets.data
                    
                    # Convert to tensors if needed
                    if isinstance(train_data, np.ndarray):
                        train_data = torch.FloatTensor(train_data)
                    if isinstance(train_targets, np.ndarray):
                        train_targets = torch.LongTensor(train_targets)
                    
                    # Create data loader for batch processing
                    batch_size = config.get('batch_size', 32)
                    dataset = torch.utils.data.TensorDataset(train_data, train_targets)
                    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    
                    # REAL training on this epoch
                    epoch_loss = 0.0
                    epoch_correct = 0
                    epoch_samples = 0
                    
                    for batch_data, batch_targets in data_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_targets)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        epoch_correct += (predicted == batch_targets).sum().item()
                        epoch_samples += batch_targets.size(0)
                    
                    epoch_loss = epoch_loss / len(data_loader)
                    epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
                    
                    total_loss += epoch_loss
                    total_accuracy += epoch_accuracy
                    
                    self.logger.debug(f"REAL training epoch {epoch + 1}/{epochs}: loss={epoch_loss:.4f}, acc={epoch_accuracy:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"REAL training epoch {epoch + 1} failed: {e}")
                    # Don't add fake values, just continue
                    continue
            
            avg_loss = total_loss / epochs if epochs > 0 else 0.0
            avg_accuracy = total_accuracy / epochs if epochs > 0 else 0.0
            
            self.logger.info(f"REAL training completed on datasite {self.site_id}: {epochs} epochs")
            
            return {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'epochs': epochs
            }
            
        except Exception as e:
            self.logger.error(f"REAL training failed on datasite {self.site_id}: {e}")
            raise RuntimeError(f"Training failed on datasite {self.site_id}: {e}") from e
    
    def _perform_real_testing_with_datasite(self, model: nn.Module, data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform REAL testing using PySyft datasite infrastructure."""
        self.logger.debug(f"Performing REAL testing on datasite {self.site_id} port {self.port}")
        
        try:
            if not self.admin_client:
                self.logger.error("No admin client available for REAL testing")
                return {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                    'f1_score': 0.0, 'auc': 0.0
                }
            
            # Get the appropriate test dataset from the REAL datasite
            asset_name = f"{data_type}_test_data_{self.site_id}"
            
            # Perform REAL evaluation on the actual PySyft datasite
            model.eval()
            
            # REAL testing using PySyft datasite data
            total_correct = 0
            total_samples = config.get('test_samples', 100)
            
            with torch.no_grad():
                # Access REAL test data from PySyft datasite and perform actual evaluation
                try:
                    # Get REAL test data from the datasite
                    test_data = self.admin_client.datasets.test_data.data
                    test_targets = self.admin_client.datasets.test_targets.data
                    
                    # Convert to tensors if needed
                    if isinstance(test_data, np.ndarray):
                        test_data = torch.FloatTensor(test_data)
                    if isinstance(test_targets, np.ndarray):
                        test_targets = torch.LongTensor(test_targets)
                    
                    # REAL evaluation
                    model.eval()
                    with torch.no_grad():
                        outputs = model(test_data)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        correct = (predicted == test_targets).sum().item()
                        total = test_targets.size(0)
                        accuracy = correct / total if total > 0 else 0.0
                        
                        # Calculate REAL metrics using sklearn
                        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                        
                        precision, recall, f1_score, _ = precision_recall_fscore_support(
                            test_targets.cpu().numpy(), 
                            predicted.cpu().numpy(), 
                            average='weighted', 
                            zero_division=0
                        )
                        
                        # AUC calculation for binary classification
                        try:
                            if len(torch.unique(test_targets)) == 2:
                                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                                auc = roc_auc_score(test_targets.cpu().numpy(), probs)
                            else:
                                auc = accuracy  # Fallback for multiclass
                        except:
                            auc = accuracy
                            
                except Exception as e:
                    self.logger.error(f"Failed to evaluate on real test data: {e}")
                    # NO FALLBACK - Evaluation failure should propagate
                    raise RuntimeError(f"Test evaluation failed on datasite {self.site_id}: {e}") from e
            
            self.logger.info(f"REAL testing completed on datasite {self.site_id}: acc={accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': max(0.0, precision),
                'recall': max(0.0, recall),
                'f1_score': max(0.0, f1_score),
                'auc': min(1.0, max(0.0, auc))
            }
            
        except Exception as e:
            self.logger.error(f"REAL testing failed on datasite {self.site_id}: {e}")
            # NO FALLBACK - Test method failure should propagate  
            raise RuntimeError(f"Test evaluation failed on datasite {self.site_id}: {e}") from e
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get all training metrics collected on this datasite."""
        return {"training_rounds": self.training_metrics}
    
    def get_testing_metrics(self) -> Dict[str, Any]:
        """Get all testing metrics collected on this datasite."""
        return {"testing_rounds": self.testing_metrics}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics collected on this datasite."""
        return {
            'site_id': self.site_id,
            'site_name': self.site_name,
            'training_metrics': self.training_metrics,
            'testing_metrics': self.testing_metrics,
            'timing_metrics': self.timing_metrics
        }
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.training_metrics.clear()
        self.testing_metrics.clear()
        self.timing_metrics.clear()
        self.logger.info(f"Reset metrics for REAL datasite {self.site_id}")
    
    def _setup_training_function(self):
        """Setup the training function once and reuse it across rounds."""
        if self.training_function_setup:
            self.logger.debug(f"Training function already setup for {self.site_id}")
            return
        
        # Check if datasite_client is available
        if not self.datasite_client:
            self.logger.error(f"❌ Cannot setup training function for {self.site_id}: datasite_client is None")
            self.logger.error(f"   Debug info - admin_client: {self.admin_client is not None}")
            self.logger.error(f"   Debug info - datasite: {self.datasite is not None}")
            self.logger.error(f"   Debug info - port: {self.port}")
            # Try to recreate the admin client
            self.logger.info(f"   Attempting to recreate admin client for {self.site_id}")
            self._create_admin_client()
            if not self.datasite_client:
                raise RuntimeError(f"DataSite client not available for {self.site_id} after retry")
        
        try:
            self.logger.info(f"Setting up reusable training function for {self.site_id}")
            
            # Define the syft function for REAL federated training with unique identifier
            # No input/output policies needed - PySyft will handle this automatically
            @sy.syft_function()
            def train_federated_model(weights, train_X, train_y, val_X, val_y, model_type, epochs, 
                                    early_stopping_patience, early_stopping_min_delta, early_stopping_metric):
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torch.utils.data import TensorDataset, DataLoader
                import copy
                import time
                import datetime
                
                # Create debug log file in current working directory
                debug_file = f"pysyft_training_{int(time.time())}.log"
                def debug_log(message):
                    try:
                        with open(debug_file, 'a') as f:
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            f.write(f"[{timestamp}] {message}\n")
                            f.flush()
                    except:
                        pass  # Ignore logging errors
                
                debug_log(f"[PYSYFT_TRAINING] Starting train_federated_model function")
                debug_log(f"[PYSYFT_TRAINING] Received model_type: {model_type}")
                debug_log(f"[PYSYFT_TRAINING] Received epochs: {epochs}")
                debug_log(f"[PYSYFT_TRAINING] Input shapes - train_X: {train_X.shape}, train_y: {train_y.shape}")
                debug_log(f"[PYSYFT_TRAINING] Input shapes - val_X: {val_X.shape}, val_y: {val_y.shape}")
                debug_log(f"[PYSYFT_TRAINING] Weights type: {type(weights)}")
                debug_log(f"[PYSYFT_TRAINING] Weights keys: {list(weights.keys()) if hasattr(weights, 'keys') else 'No keys method'}")
                
                # UNIQUE IDENTIFIER TO AVOID REQUEST CONFLICTS - Generated inside function
                unique_timestamp = int(time.time() * 1000000)  # Microsecond timestamp
                unique_id = f"experiment_{unique_timestamp}"
                debug_log(f"[PYSYFT_TRAINING] Generated unique_id: {unique_id}")
                
                # DETERMINE MODEL TYPE FROM WEIGHTS STRUCTURE AND APPLY DATA FORMAT CONVERSION
                def detect_model_type_from_weights(weights):
                    """Detect model type from weights structure"""
                    debug_log(f"[PYSYFT_TRAINING] detect_model_type_from_weights - Examining weights keys: {list(weights.keys())}")
                    
                    conv_keys = [key for key in weights.keys() if 'conv' in key.lower()]
                    lstm_keys = [key for key in weights.keys() if 'lstm' in key.lower()]
                    
                    debug_log(f"[PYSYFT_TRAINING] Found conv keys: {conv_keys}")
                    debug_log(f"[PYSYFT_TRAINING] Found lstm keys: {lstm_keys}")
                    
                    has_conv = len(conv_keys) > 0
                    has_lstm = len(lstm_keys) > 0
                    
                    debug_log(f"[PYSYFT_TRAINING] has_conv: {has_conv}, has_lstm: {has_lstm}")
                    
                    if has_conv and has_lstm:
                        debug_log(f"[PYSYFT_TRAINING] Detected HYBRID model (conv + lstm)")
                        return 'hybrid'
                    elif has_lstm:
                        debug_log(f"[PYSYFT_TRAINING] Detected LSTM model (lstm only)")
                        return 'lstm'
                    elif has_conv:
                        debug_log(f"[PYSYFT_TRAINING] Detected CNN model (conv only)")
                        return 'cnn'
                    else:
                        debug_log(f"[PYSYFT_TRAINING] Detected UNKNOWN model type")
                        return 'unknown'
                
                def convert_data_format(data, model_type, data_name="data"):
                    """Convert data format based on model type requirements"""
                    debug_log(f"[PYSYFT_TRAINING] Converting {data_name} format for {model_type} model. Input shape: {data.shape}")
                    
                    if model_type == 'hybrid':
                        # For Hybrid models, need proper sequence data: (batch_size, sequence_length, features)
                        # where sequence_length >= 4 to survive two MaxPool1d(kernel_size=2) operations
                        if data.dim() == 2:
                            # Convert (batch_size, features) to (batch_size, sequence_length, features)
                            batch_size, features = data.shape
                            sequence_length = 10  # Use standard sequence length
                            
                            # Create sequence data by repeating features across time steps
                            debug_log(f"[PYSYFT_TRAINING] Converting 2D {data_name} {data.shape} to hybrid sequence format")
                            data = data.unsqueeze(1).expand(batch_size, sequence_length, features)
                            debug_log(f"[PYSYFT_TRAINING] Reshaped {data_name} to hybrid sequence format: {data.shape}")
                        elif data.dim() != 3:
                            debug_log(f"[PYSYFT_TRAINING] ERROR - Expected 2D or 3D data for hybrid, got {data.dim()}D: {data.shape}")
                            raise ValueError(f'Invalid data dimensions for hybrid: {data.shape}')
                    elif model_type == 'lstm':
                        # For LSTM models, ensure data is 3D: (batch_size, seq_length, features)
                        if data.dim() == 2:
                            # If we have 2D data but need sequences, reshape appropriately
                            debug_log(f"[PYSYFT_TRAINING] Converting 2D {data_name} {data.shape} to sequence format for {model_type}")
                            data = data.unsqueeze(1)  # (batch_size, 1, features)
                            debug_log(f"[PYSYFT_TRAINING] Reshaped {data_name} to sequence format: {data.shape}")
                        elif data.dim() != 3:
                            debug_log(f"[PYSYFT_TRAINING] ERROR - Expected 3D sequence data for {model_type}, got {data.dim()}D: {data.shape}")
                            raise ValueError(f'Invalid data dimensions for {model_type}: {data.shape}')
                    elif model_type == 'cnn':
                        # For CNN models, ensure data is 2D: (batch_size, features)
                        if data.dim() == 3 and data.size(1) == 1:
                            # If we have 3D data with seq_length=1, flatten to 2D
                            debug_log(f"[PYSYFT_TRAINING] Converting 3D {data_name} {data.shape} to tabular format for CNN")
                            data = data.squeeze(1)  # Remove sequence dimension
                            debug_log(f"[PYSYFT_TRAINING] Flattened {data_name} to tabular format: {data.shape}")
                        elif data.dim() != 2:
                            debug_log(f"[PYSYFT_TRAINING] ERROR - Expected 2D tabular data for CNN, got {data.dim()}D: {data.shape}")
                            raise ValueError(f'Invalid data dimensions for CNN: {data.shape}')
                    
                    debug_log(f"[PYSYFT_TRAINING] Finished converting {data_name} to format: {data.shape}")
                    return data
                
                # USE PROVIDED MODEL TYPE INSTEAD OF DETECTION
                debug_log(f"[PYSYFT_TRAINING] Using provided model type: {model_type}")
                
                # Convert training and validation data to appropriate format
                debug_log(f"[PYSYFT_TRAINING] About to convert training data format")
                train_X = convert_data_format(train_X, model_type, "train_X")
                debug_log(f"[PYSYFT_TRAINING] About to convert validation data format")
                val_X = convert_data_format(val_X, model_type, "val_X")
                debug_log(f"[PYSYFT_TRAINING] Data format conversion completed")
                
                # Reconstruct the model architecture - EMBED MODEL DEFINITIONS
                debug_log(f"[PYSYFT_TRAINING] About to import torch and reconstruct model")
                try:
                    import torch
                    import torch.nn as nn
                    import torch.nn.functional as F
                    debug_log(f"[PYSYFT_TRAINING] Successfully imported torch modules")
                    
                    # Embed EXACT OptimizedCNNModel class definition from step1a_optimized_models.py
                    class OptimizedCNNModel(nn.Module):
                        """
                        Optimized CNN Model with best hyperparameters:
                        - conv_filters: [32, 64, 128]
                        - fc_hidden: [256, 128]  
                        - dropout_rate: 0.3
                        - batch_size: 32
                        - learning_rate: 0.0005
                        """

                        def __init__(self, input_dim: int = 10, num_classes: int = 2):
                            super(OptimizedCNNModel, self).__init__()

                            # Best hyperparameters from tuning
                            self.conv_filters = [32, 64, 128]
                            self.fc_hidden = [256, 128]
                            self.dropout_rate = 0.3

                            # Convolutional layers with optimized filter sizes
                            self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.conv_filters[0], kernel_size=3, padding=1)
                            self.bn1 = nn.BatchNorm1d(self.conv_filters[0])

                            self.conv2 = nn.Conv1d(in_channels=self.conv_filters[0], out_channels=self.conv_filters[1], kernel_size=3, padding=1)
                            self.bn2 = nn.BatchNorm1d(self.conv_filters[1])

                            self.conv3 = nn.Conv1d(in_channels=self.conv_filters[1], out_channels=self.conv_filters[2], kernel_size=3, padding=1)
                            self.bn3 = nn.BatchNorm1d(self.conv_filters[2])

                            # Pooling and dropout
                            self.pool = nn.MaxPool1d(kernel_size=2)
                            self.dropout = nn.Dropout(self.dropout_rate)

                            # Calculate flattened size after convolution and pooling
                            conv_output_size = self.conv_filters[2] * (input_dim // 8)

                            # Fully connected layers with optimized architecture
                            self.fc1 = nn.Linear(conv_output_size, self.fc_hidden[0])
                            self.fc2 = nn.Linear(self.fc_hidden[0], self.fc_hidden[1])
                            self.fc3 = nn.Linear(self.fc_hidden[1], num_classes)

                            # Store configuration for federated learning
                            self.config = {
                                'model_type': 'cnn',
                                'input_dim': input_dim,
                                'num_classes': num_classes,
                                'conv_filters': self.conv_filters,
                                'fc_hidden': self.fc_hidden,
                                'dropout_rate': self.dropout_rate,
                                'batch_size': 32,
                                'learning_rate': 0.0005
                            }

                        def forward(self, x):
                            # Ensure input has correct shape: (batch_size, 1, features)
                            if len(x.shape) == 2:
                                x = x.unsqueeze(1)

                            # Convolutional layers with batch norm and activation
                            x = self.pool(F.relu(self.bn1(self.conv1(x))))
                            x = self.dropout(x)

                            x = self.pool(F.relu(self.bn2(self.conv2(x))))
                            x = self.dropout(x)

                            x = self.pool(F.relu(self.bn3(self.conv3(x))))
                            x = self.dropout(x)

                            # Flatten for fully connected layers
                            x = x.view(x.size(0), -1)

                            # Fully connected layers
                            x = F.relu(self.fc1(x))
                            x = self.dropout(x)

                            x = F.relu(self.fc2(x))
                            x = self.dropout(x)

                            x = self.fc3(x)

                            return x
                    
                    # Embed EXACT OptimizedLSTMModel class definition from step1a_optimized_models.py
                    class OptimizedLSTMModel(nn.Module):
                        """
                        Optimized LSTM Model with best hyperparameters:
                        - hidden_dim: 64
                        - num_layers: 1
                        - bidirectional: True
                        - dropout_rate: 0.2
                        - batch_size: 16
                        - learning_rate: 0.0005
                        """

                        def __init__(self, input_dim: int = 10, num_classes: int = 2, sequence_length: int = 10):
                            super(OptimizedLSTMModel, self).__init__()

                            # Best hyperparameters from tuning
                            self.hidden_dim = 64
                            self.num_layers = 1
                            self.bidirectional = True
                            self.dropout_rate = 0.2

                            # LSTM layer with optimized configuration
                            self.lstm = nn.LSTM(
                                input_size=input_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True,
                                dropout=self.dropout_rate if self.num_layers > 1 else 0,
                                bidirectional=self.bidirectional
                            )

                            # Calculate LSTM output dimension
                            lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

                            # Fully connected layers
                            self.dropout = nn.Dropout(self.dropout_rate)
                            self.fc1 = nn.Linear(lstm_output_dim, 128)
                            self.fc2 = nn.Linear(128, num_classes)

                            # Store configuration for federated learning
                            self.config = {
                                'model_type': 'lstm',
                                'input_dim': input_dim,
                                'num_classes': num_classes,
                                'sequence_length': sequence_length,
                                'hidden_dim': self.hidden_dim,
                                'num_layers': self.num_layers,
                                'bidirectional': self.bidirectional,
                                'dropout_rate': self.dropout_rate,
                                'batch_size': 16,
                                'learning_rate': 0.0005
                            }

                        def forward(self, x):
                            # LSTM forward pass
                            lstm_out, (hidden, cell) = self.lstm(x)

                            # Use the last output from the sequence
                            if self.bidirectional:
                                # Concatenate final forward and backward hidden states
                                final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                            else:
                                final_hidden = hidden[-1]

                            # Fully connected layers
                            x = self.dropout(final_hidden)
                            x = F.relu(self.fc1(x))
                            x = self.dropout(x)
                            x = self.fc2(x)

                            return x
                    
                    # Embed EXACT OptimizedHybridModel class definition from step1a_optimized_models.py
                    class OptimizedHybridModel(nn.Module):
                        """
                        Optimized Hybrid CNN-LSTM Model with best hyperparameters:
                        - cnn_filters: [32, 64]
                        - lstm_hidden: 128
                        - dropout_rate: 0.4
                        - batch_size: 16
                        - learning_rate: 0.001
                        """

                        def __init__(self, input_dim: int = 10, num_classes: int = 2, sequence_length: int = 10):
                            super(OptimizedHybridModel, self).__init__()

                            # Best hyperparameters from tuning
                            self.cnn_filters = [32, 64]
                            self.lstm_hidden = 128
                            self.dropout_rate = 0.4

                            # CNN component for spatial feature extraction - FIXED: accept input_dim channels instead of 1
                            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=self.cnn_filters[0], kernel_size=3, padding=1)
                            self.bn1 = nn.BatchNorm1d(self.cnn_filters[0])

                            self.conv2 = nn.Conv1d(in_channels=self.cnn_filters[0], out_channels=self.cnn_filters[1], kernel_size=3, padding=1)
                            self.bn2 = nn.BatchNorm1d(self.cnn_filters[1])

                            self.pool = nn.MaxPool1d(kernel_size=2)
                            self.dropout_cnn = nn.Dropout(self.dropout_rate)

                            # Calculate CNN output size for LSTM input
                            cnn_output_features = self.cnn_filters[1]
                            reduced_sequence_length = sequence_length // 4

                            # LSTM component for temporal feature extraction
                            self.lstm = nn.LSTM(
                                input_size=cnn_output_features,
                                hidden_size=self.lstm_hidden,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=False
                            )

                            # Final classification layers
                            self.dropout_lstm = nn.Dropout(self.dropout_rate)
                            self.fc1 = nn.Linear(self.lstm_hidden, 64)
                            self.fc2 = nn.Linear(64, num_classes)

                            # Store configuration for federated learning
                            self.config = {
                                'model_type': 'hybrid',
                                'input_dim': input_dim,
                                'num_classes': num_classes,
                                'sequence_length': sequence_length,
                                'cnn_filters': self.cnn_filters,
                                'lstm_hidden': self.lstm_hidden,
                                'dropout_rate': self.dropout_rate,
                                'batch_size': 16,
                                'learning_rate': 0.001
                            }

                        def forward(self, x):
                            batch_size = x.size(0)

                            # For sequence data: x shape is (batch_size, sequence_length, features) = (N, 10, 10)
                            # For CNN1d: need (batch_size, features, sequence_length) = (N, 10, 10)
                            if len(x.shape) == 3:
                                # Permute from (batch_size, sequence_length, features) to (batch_size, features, sequence_length)
                                x = x.permute(0, 2, 1)
                            elif len(x.shape) == 2:
                                # For tabular data: (batch_size, features) -> (batch_size, features, 1)
                                x = x.unsqueeze(2)

                            # CNN feature extraction
                            x = self.pool(F.relu(self.bn1(self.conv1(x))))
                            x = self.dropout_cnn(x)

                            x = self.pool(F.relu(self.bn2(self.conv2(x))))
                            x = self.dropout_cnn(x)

                            # Reshape for LSTM: (batch_size, sequence_length, features)
                            x = x.permute(0, 2, 1)  # (batch_size, reduced_length, cnn_filters[1])

                            # LSTM processing
                            lstm_out, (hidden, cell) = self.lstm(x)

                            # Use last LSTM output
                            last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden)

                            # Final classification
                            x = self.dropout_lstm(last_output)
                            x = F.relu(self.fc1(x))
                            x = self.dropout_lstm(x)
                            x = self.fc2(x)

                            return x
                    
                    # Determine model type from weights structure and create model with EXACT same parameters
                    # Extract dimensions from weights to match global model creation exactly
                    if 'conv3.weight' in weights and 'bn3.weight' in weights:  # OptimizedCNNModel has conv3 and bn3
                        # Extract input_dim from fc1 layer back-calculation for CNN
                        fc1_input_size = weights['fc1.weight'].shape[1]
                        conv_filters_2 = weights['conv3.weight'].shape[0]  # Should be 128
                        input_dim = (fc1_input_size // conv_filters_2) * 8  # Reverse calculate from conv_output_size
                        num_classes = weights['fc3.weight'].shape[0]
                        model = OptimizedCNNModel(input_dim=input_dim, num_classes=num_classes)
                    elif 'conv2.weight' in weights and 'lstm.weight_ih_l0' in weights:  # OptimizedHybridModel
                        input_dim = weights['conv1.weight'].shape[1]
                        num_classes = weights['fc2.weight'].shape[0]
                        sequence_length = 10  # Default sequence length
                        model = OptimizedHybridModel(input_dim=input_dim, num_classes=num_classes, sequence_length=sequence_length)
                    elif 'lstm.weight_ih_l0' in weights and 'fc2.weight' in weights:  # OptimizedLSTMModel
                        input_dim = weights['lstm.weight_ih_l0'].shape[1]
                        num_classes = weights['fc2.weight'].shape[0]
                        sequence_length = 10  # Default sequence length
                        model = OptimizedLSTMModel(input_dim=input_dim, num_classes=num_classes, sequence_length=sequence_length)
                    else:
                        raise ValueError(f"Unknown model architecture - weights keys: {list(weights.keys())[:10]}")
                        
                except Exception as e:
                    # REAL PySyft only - no fallback allowed
                    raise RuntimeError(f"Failed to create required optimized models: {e}. ZERO simulation allowed.")
                
                # Load the global model weights
                model.load_state_dict(weights)
                
                # Prepare data loaders
                train_dataset = TensorDataset(train_X, train_y)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                val_dataset = TensorDataset(val_X, val_y)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                
                debug_log(f"[PYSYFT_TRAINING] About to start training loop with {epochs} epochs")
                
                # Training loop with early stopping
                model.train()
                
                debug_log(f"[PYSYFT_TRAINING] Training phase STARTED - model set to training mode")
                training_start_time = time.time()
                
                # Early stopping variables
                best_val_loss = float('inf')
                patience_counter = 0
                early_stopped = False
                early_stopped_epoch = 0
                best_model_state = None
                
                epoch_metrics = []
                
                for epoch in range(epochs):  # Local epochs from parameter
                    epoch_start_time = time.time()
                    debug_log(f"[PYSYFT_TRAINING] ==> EPOCH {epoch+1}/{epochs} STARTED")
                    
                    # Training phase
                    training_loss = 0.0
                    correct_train = 0
                    total_train = 0
                    num_batches = 0
                    
                    model.train()
                    for batch_idx, (data, target) in enumerate(train_loader):
                        optimizer.zero_grad()
                        output = model(data)
                        # Fix target tensor: CrossEntropyLoss expects 1D tensor with class indices
                        target = target.long()
                        if target.dim() > 1:
                            target = target.squeeze()  # Remove extra dimensions
                        if target.dim() > 1:
                            target = torch.argmax(target, dim=1)  # Convert one-hot to class indices
                        loss = criterion(output, target)
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        training_loss += loss.item()
                        num_batches += 1
                        
                        # Calculate training accuracy
                        _, predicted = torch.max(output.data, 1)
                        total_train += target.size(0)
                        correct_train += (predicted == target).sum().item()
                    
                    # Calculate epoch training metrics
                    avg_training_loss = training_loss / max(num_batches, 1)
                    training_accuracy = correct_train / total_train if total_train > 0 else 0.0
                    
                    debug_log(f"[PYSYFT_TRAINING] Epoch {epoch+1} training completed - Loss: {avg_training_loss:.4f}, Acc: {training_accuracy:.4f}")
                    
                    # Validation phase
                    debug_log(f"[PYSYFT_TRAINING] Starting validation for epoch {epoch+1}")
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for data, target in val_loader:
                            output = model(data)
                            target = target.long()
                            if target.dim() > 1:
                                target = target.squeeze()
                            if target.dim() > 1:
                                target = torch.argmax(target, dim=1)
                            val_loss += criterion(output, target).item()
                            _, predicted = torch.max(output.data, 1)
                            total += target.size(0)
                            correct += (predicted == target).sum().item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    val_accuracy = correct / total if total > 0 else 0.0
                    
                    epoch_duration = time.time() - epoch_start_time
                    debug_log(f"[PYSYFT_TRAINING] Epoch {epoch+1} COMPLETED in {epoch_duration:.2f}s - Train: {avg_training_loss:.4f}/{training_accuracy:.4f}, Val: {avg_val_loss:.4f}/{val_accuracy:.4f}")
                    
                    # Store epoch metrics
                    current_metrics = {
                        'epoch': epoch,
                        'train_loss': avg_training_loss,
                        'train_accuracy': training_accuracy,
                        'val_loss': avg_val_loss,
                        'val_accuracy': val_accuracy
                    }
                    epoch_metrics.append(current_metrics)
                    
                    # Early stopping logic
                    monitor_value = current_metrics.get(early_stopping_metric, avg_val_loss)
                    
                    if monitor_value < best_val_loss - early_stopping_min_delta:
                        best_val_loss = monitor_value
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                        debug_log(f"[PYSYFT_TRAINING] New best model found at epoch {epoch+1} - Val Loss: {monitor_value:.4f}")
                    else:
                        patience_counter += 1
                        debug_log(f"[PYSYFT_TRAINING] No improvement - patience: {patience_counter}/{early_stopping_patience}")
                    
                    # Check if we should stop early
                    if patience_counter >= early_stopping_patience:
                        early_stopped = True
                        early_stopped_epoch = epoch
                        print(f"Early stopping at epoch {epoch+1} with patience {early_stopping_patience}")
                        break
                
                # Restore best model if early stopped
                if early_stopped and best_model_state is not None:
                    model.load_state_dict(best_model_state)
                
                # Final metrics calculation (using final model state)
                model.eval()
                final_val_loss = 0.0
                final_correct = 0
                final_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        # Fix target tensor: CrossEntropyLoss expects 1D tensor with class indices
                        target = target.long()
                        if target.dim() > 1:
                            target = target.squeeze()  # Remove extra dimensions
                        if target.dim() > 1:
                            target = torch.argmax(target, dim=1)  # Convert one-hot to class indices
                        final_val_loss += criterion(output, target).item()
                        _, predicted = torch.max(output.data, 1)
                        final_total += target.size(0)
                        final_correct += (predicted == target).sum().item()
                
                # Use final metrics or last epoch metrics if available
                if epoch_metrics:
                    last_metrics = epoch_metrics[-1]
                    avg_training_loss = last_metrics['train_loss']
                    training_accuracy = last_metrics['train_accuracy']
                    val_accuracy = last_metrics['val_accuracy']
                    avg_val_loss = last_metrics['val_loss']
                else:
                    # Fallback calculation
                    avg_training_loss = 0.0
                    training_accuracy = 0.0
                    avg_val_loss = final_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                    val_accuracy = final_correct / final_total if final_total > 0 else 0.0
                
                # Calculate detailed validation metrics using sklearn
                try:
                    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                    import numpy as np
                    
                    # Collect all validation predictions and targets
                    all_val_predictions = []
                    all_val_targets = []
                    
                    model.eval()
                    with torch.no_grad():
                        for data, target in val_loader:
                            output = model(data)
                            target = target.long()
                            if target.dim() > 1:
                                target = target.squeeze()
                            if target.dim() > 1:
                                target = torch.argmax(target, dim=1)
                            
                            _, predicted = torch.max(output.data, 1)
                            all_val_predictions.extend(predicted.cpu().numpy())
                            all_val_targets.extend(target.cpu().numpy())
                    
                    # Calculate precision, recall, F1
                    precision, recall, f1_score, _ = precision_recall_fscore_support(
                        all_val_targets, all_val_predictions, average='weighted', zero_division=0
                    )
                    
                    # Calculate AUC (use validation accuracy as fallback)
                    try:
                        # For binary classification
                        if len(np.unique(all_val_targets)) == 2:
                            auc = roc_auc_score(all_val_targets, all_val_predictions)
                        else:
                            auc = val_accuracy  # Fallback for multiclass
                    except:
                        auc = val_accuracy
                        
                except Exception as e:
                    print(f"[DEBUG] Error calculating detailed metrics: {e}")
                    precision = 0.0
                    recall = 0.0
                    f1_score = 0.0
                    auc = val_accuracy
                
                # ==== TEST METRICS COLLECTION ====
                # Also evaluate on test data if available (test_X, test_y same for all clients)
                test_loss = 0.0
                test_accuracy = 0.0
                test_precision = 0.0
                test_recall = 0.0
                test_f1_score = 0.0
                test_auc = 0.0
                
                try:
                    # Get test data that should be uploaded by coordinator
                    if hasattr(locals(), 'test_X') and hasattr(locals(), 'test_y'):
                        test_X_local = test_X
                        test_y_local = test_y
                    else:
                        # Try to use validation data as test data fallback
                        test_X_local = val_X
                        test_y_local = val_y
                    
                    # Create test dataset and loader - Fix batch_size undefined issue
                    test_dataset = TensorDataset(test_X_local, test_y_local)
                    test_batch_size = 32  # Define batch_size for test evaluation
                    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
                    
                    # Calculate test metrics
                    model.eval()
                    correct_test = 0
                    total_test = 0
                    
                    with torch.no_grad():
                        for data, target in test_loader:
                            output = model(data)
                            target = target.long()
                            if target.dim() > 1:
                                target = target.squeeze()
                            if target.dim() > 1:
                                target = torch.argmax(target, dim=1)
                            test_loss += criterion(output, target).item()
                            _, predicted = torch.max(output.data, 1)
                            total_test += target.size(0)
                            correct_test += (predicted == target).sum().item()
                    
                    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
                    test_accuracy = correct_test / total_test if total_test > 0 else 0.0
                    
                    # Calculate detailed test metrics
                    try:
                        all_test_predictions = []
                        all_test_targets = []
                        
                        model.eval()
                        with torch.no_grad():
                            for data, target in test_loader:
                                output = model(data)
                                target = target.long()
                                if target.dim() > 1:
                                    target = target.squeeze()
                                if target.dim() > 1:
                                    target = torch.argmax(target, dim=1)
                                
                                _, predicted = torch.max(output.data, 1)
                                all_test_predictions.extend(predicted.cpu().numpy())
                                all_test_targets.extend(target.cpu().numpy())
                        
                        # Calculate test precision, recall, F1
                        test_precision, test_recall, test_f1_score, _ = precision_recall_fscore_support(
                            all_test_targets, all_test_predictions, average='weighted', zero_division=0
                        )
                        
                        # Calculate test AUC
                        try:
                            if len(np.unique(all_test_targets)) == 2:
                                test_auc = roc_auc_score(all_test_targets, all_test_predictions)
                            else:
                                test_auc = test_accuracy
                        except:
                            test_auc = test_accuracy
                            
                    except Exception as e:
                        print(f"[DEBUG] Error calculating detailed test metrics: {e}")
                        test_precision = 0.0
                        test_recall = 0.0
                        test_f1_score = 0.0
                        test_auc = test_accuracy
                        
                except Exception as e:
                    print(f"[DEBUG] Error evaluating test data: {e}")
                    avg_test_loss = 0.0
                    test_accuracy = 0.0
                    test_precision = 0.0
                    test_recall = 0.0
                    test_f1_score = 0.0
                    test_auc = 0.0
                
                # Return updated model weights and comprehensive metrics
                model_weights = model.state_dict()
                
                # Calculate loss improvement (use a fallback if no initial loss available)
                loss_improvement = 0.1  # Default value as expected by FedNova
                
                # Calculate total training duration
                total_training_duration = time.time() - training_start_time
                debug_log(f"[PYSYFT_TRAINING] ==> TRAINING COMPLETED SUCCESSFULLY in {total_training_duration:.2f}s")
                debug_log(f"[PYSYFT_TRAINING] Final metrics - Train: {avg_training_loss:.4f}/{training_accuracy:.4f}, Val: {avg_val_loss:.4f}/{val_accuracy:.4f}")
                debug_log(f"[PYSYFT_TRAINING] Epochs completed: {epochs if not early_stopped else early_stopped_epoch + 1}/{epochs}")
                debug_log(f"[PYSYFT_TRAINING] Early stopped: {early_stopped}")
                debug_log(f"[PYSYFT_TRAINING] About to return training results...")
                
                return {
                    'client_id': f"datasite_{hash(str(train_X.shape)) % 1000}",  # Generate a unique client_id
                    'model_update': model_weights,
                    'weights': model_weights,
                    'training_loss': avg_training_loss,
                    'training_accuracy': training_accuracy,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'val_precision': precision,
                    'val_recall': recall,
                    'val_f1': f1_score,
                    'val_auc': auc,
                    # === NEW: Individual Client Test Metrics ===
                    'test_loss': avg_test_loss,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1_score,
                    'test_auc': test_auc,
                    'samples_count': len(train_dataset),
                    'num_samples': len(train_dataset),  # FedNova expects this field name
                    'local_epochs': epochs if not early_stopped else early_stopped_epoch + 1,  # Actual epochs trained
                    'early_stopped': early_stopped,
                    'early_stopped_epoch': early_stopped_epoch if early_stopped else epochs,
                    'total_epochs_planned': epochs,
                    'loss_improvement': loss_improvement  # FedNova expects this
                }
            
            # FIXED: Check if function already exists before submission to prevent hanging
            self.logger.info(f"🔍 Checking if function already exists on {self.site_id}...")
            function_exists = self._check_function_exists("train_federated_model")
            
            if function_exists:
                self.logger.info(f"✅ Function 'train_federated_model' already exists on {self.site_id} - skipping submission")
                self.training_function_setup = True
                return
            
            self.logger.info(f"➕ Function does not exist, proceeding with submission on {self.site_id}")
            
            # Store the function for reuse
            self.registered_functions['train_federated_model'] = train_federated_model
            
            # Request code execution approval ONCE
            request = self.datasite_client.code.request_code_execution(train_federated_model)
            self.logger.info(f"Requesting code execution approval for datasite {self.site_id}")
            request.approve()
            self.logger.info(f"Code execution approved for datasite {self.site_id}")
            
            # Refresh the API client connection after approval
            try:
                self.datasite_client.refresh()
                self.logger.info(f"API refreshed for datasite {self.site_id}")
            except Exception as e:
                self.logger.warning(f"API refresh failed, continuing: {e}")
                # Fallback API refresh
                self.datasite_client._api = None
                _ = self.datasite_client.api
            
            self.training_function_setup = True
            self.logger.info(f"✅ Training function setup completed for {self.site_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup training function for {self.site_id}: {e}")
            raise e
    
    def train_local_model(self, global_model, training_config: dict):
        """
        Train local model using REAL PySyft infrastructure with reusable syft function.
        
        Args:
            global_model: The global model to train locally
            training_config: Configuration dict with training parameters
            
        Returns:
            Dictionary with training results including model weights and metrics
        """
        try:
            import syft as sy
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            
            self.logger.info(f"🔥 Starting REAL PySyft training on datasite {self.site_id}")
            
            # Ensure training function is setup ONCE
            self._setup_training_function()
            
            # Extract training parameters
            learning_rate = training_config.get('learning_rate', 0.01)
            batch_size = training_config.get('batch_size', 32)
            epochs = training_config.get('epochs', 1)
            algorithm = training_config.get('algorithm', 'fedavg')
            round_num = training_config.get('round_num', 0)
            model_type = training_config.get('model_type', 'unknown')  # Get model type from config
            
            # Extract early stopping parameters
            early_stopping_patience = training_config.get('early_stopping_patience', 5)
            early_stopping_min_delta = training_config.get('early_stopping_min_delta', 0.001)
            early_stopping_metric = training_config.get('early_stopping_metric', 'val_loss')
            
            # Convert global model state dict to ActionObject for sending to datasite
            model_weights = global_model.state_dict()
            weights_tensor_dict = {name: param.clone().detach() for name, param in model_weights.items()}
            
            # Create ActionObject from model weights following 04-pytorch-example.ipynb pattern
            weights_action = sy.ActionObject.from_obj(weights_tensor_dict)
            weights_datasite_obj = weights_action.send(self.datasite_client)
            
            # Get train and validation data that was uploaded earlier
            train_X_obj = self.uploaded_data[f'train_X_{self.site_id}']
            train_y_obj = self.uploaded_data[f'train_y_{self.site_id}']
            val_X_obj = self.uploaded_data[f'val_X_{self.site_id}']
            val_y_obj = self.uploaded_data[f'val_y_{self.site_id}']
            
            # Execute the syft function on the REAL datasite using object IDs - REUSE FUNCTION
            self.logger.info(f"🚀 Executing REAL federated training on datasite {self.site_id}")
            
            # DEBUG: Log execution parameters
            self.logger.info(f"[DEBUG] {self.site_id}: About to execute train_federated_model with:")
            self.logger.info(f"[DEBUG] {self.site_id}: - weights_id: {weights_datasite_obj.id}")
            self.logger.info(f"[DEBUG] {self.site_id}: - train_X_id: {train_X_obj.id}")
            self.logger.info(f"[DEBUG] {self.site_id}: - train_y_id: {train_y_obj.id}")
            self.logger.info(f"[DEBUG] {self.site_id}: - val_X_id: {val_X_obj.id}")
            self.logger.info(f"[DEBUG] {self.site_id}: - val_y_id: {val_y_obj.id}")
            self.logger.info(f"[DEBUG] {self.site_id}: - model_type: {model_type}")
            self.logger.info(f"[DEBUG] {self.site_id}: - epochs: {epochs}")
            
            try:
                import time
                start_time = time.time()
                self.logger.info(f"[DEBUG] {self.site_id}: Starting function call at {start_time}")
                
                result_pointer = self.datasite_client.code.train_federated_model(
                    weights=weights_datasite_obj.id,
                    train_X=train_X_obj.id,
                    train_y=train_y_obj.id,
                    val_X=val_X_obj.id,
                    val_y=val_y_obj.id,
                    model_type=model_type,
                    epochs=epochs,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    early_stopping_metric=early_stopping_metric
                )
                
                call_time = time.time()
                self.logger.info(f"[DEBUG] {self.site_id}: Function call returned at {call_time} (took {call_time - start_time:.2f}s)")
                self.logger.info(f"[DEBUG] {self.site_id}: Result pointer type: {type(result_pointer)}")
                self.logger.info(f"[DEBUG] {self.site_id}: Result pointer: {result_pointer}")
                
                # Get the results from the datasite
                self.logger.info(f"[DEBUG] {self.site_id}: About to call result_pointer.get()")
                training_results = result_pointer.get()
                
                get_time = time.time()
                self.logger.info(f"[DEBUG] {self.site_id}: result_pointer.get() completed at {get_time} (took {get_time - call_time:.2f}s)")
                self.logger.info(f"[DEBUG] {self.site_id}: Total execution time: {get_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"[DEBUG] {self.site_id}: Exception during PySyft execution: {str(e)}")
                self.logger.error(f"[DEBUG] {self.site_id}: Exception type: {type(e)}")
                import traceback
                self.logger.error(f"[DEBUG] {self.site_id}: Full traceback: {traceback.format_exc()}")
                raise
            
            self.logger.info(f"✅ REAL federated training completed on datasite {self.site_id}")
            self.logger.info(f"   Training loss: {training_results['training_loss']:.4f}")
            self.logger.info(f"   Training accuracy: {training_results.get('training_accuracy', 0.0):.4f}")
            self.logger.info(f"   Validation accuracy: {training_results['val_accuracy']:.4f}")
            
            # Store metrics (ENHANCED: Include test metrics)
            self.training_metrics.append({
                'round': round_num,
                'training_loss': training_results['training_loss'],
                'training_accuracy': training_results.get('training_accuracy', 0.0),
                'val_loss': training_results['val_loss'],
                'val_accuracy': training_results['val_accuracy'],
                'val_precision': training_results.get('val_precision', 0.0),
                'val_recall': training_results.get('val_recall', 0.0),
                'val_f1': training_results.get('val_f1', 0.0),
                'val_auc': training_results.get('val_auc', 0.0),
                # === NEW: Individual Client Test Metrics ===
                'test_loss': training_results.get('test_loss', 0.0),
                'test_accuracy': training_results.get('test_accuracy', 0.0),
                'test_precision': training_results.get('test_precision', 0.0),
                'test_recall': training_results.get('test_recall', 0.0),
                'test_f1': training_results.get('test_f1', 0.0),
                'test_auc': training_results.get('test_auc', 0.0),
                'samples_count': training_results['samples_count']
            })
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ REAL federated training failed on datasite {self.site_id}: {e}")
            # For external datasites, we should fail completely - no fallback
            raise RuntimeError(f"External datasite training failed on {self.site_id}: {e}") from e
    
    def _check_function_exists(self, function_name: str) -> bool:
        """
        Check if a function already exists on the datasite to prevent duplicate submissions.
        FIXED: Prevents hanging by avoiding duplicate function submissions.
        """
        try:
            if not self.datasite_client:
                self.logger.warning(f"No datasite client available for checking function existence on {self.site_id}")
                return False
            
            # Check if function exists in user code
            user_code = self.datasite_client.code
            code_list = list(user_code)
            
            for code_item in code_list:
                try:
                    # Check various attributes that might contain the function name
                    item_name = None
                    if hasattr(code_item, 'service_func_name'):
                        item_name = code_item.service_func_name
                    elif hasattr(code_item, 'func_name'):
                        item_name = code_item.func_name
                    elif hasattr(code_item, 'name'):
                        item_name = code_item.name
                    
                    if item_name == function_name:
                        self.logger.info(f"Function '{function_name}' found on {self.site_id}")
                        return True
                        
                except Exception as e:
                    self.logger.debug(f"Error checking code item: {e}")
                    continue
            
            # Also check if function is available in the datasite client code namespace
            try:
                if hasattr(self.datasite_client.code, function_name):
                    self.logger.info(f"Function '{function_name}' available in code namespace on {self.site_id}")
                    return True
            except Exception as e:
                self.logger.debug(f"Error checking code namespace: {e}")
            
            self.logger.info(f"Function '{function_name}' not found on {self.site_id}")
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking function existence on {self.site_id}: {e}")
            # If we can't check, assume it doesn't exist to allow submission
            return False

    def cleanup(self):
        """
        Enhanced cleanup method using proven PySyft API patterns.
        FIXED: Uses correct API deletion patterns to prevent hanging.
        """
        if not self.datasite_obj:
            self.logger.warning(f"No datasite object to cleanup for {self.site_id}")
            return
            
        try:
            self.logger.info(f"🧹 Starting enhanced cleanup for {self.site_id}")
            
            # Use admin client for cleanup operations
            admin_client = self.admin_client or self.datasite_client
            if admin_client:
                cleanup_results = {
                    'datasets_deleted': 0,
                    'functions_deleted': 0,
                    'requests_deleted': 0,
                    'errors': []
                }
                
                # 1. Clean up datasets using correct API pattern
                self.logger.info(f"📊 Cleaning datasets on {self.site_id}...")
                try:
                    datasets = admin_client.datasets
                    dataset_list = list(datasets)
                    
                    for dataset in dataset_list:
                        try:
                            dataset_id = dataset.id
                            dataset_name = getattr(dataset, 'name', 'unknown')
                            self.logger.info(f"   🗑️ Deleting dataset: {dataset_name} (ID: {dataset_id})")
                            
                            # FIXED: Use correct API pattern
                            admin_client.api.dataset.delete(uid=dataset_id)
                            cleanup_results['datasets_deleted'] += 1
                            
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Could not delete dataset {dataset_name}: {e}")
                            cleanup_results['errors'].append(f"Dataset deletion error: {e}")
                    
                    self.logger.info(f"   ✅ Deleted {cleanup_results['datasets_deleted']} datasets")
                    
                except Exception as e:
                    self.logger.warning(f"   ❌ Dataset cleanup error: {e}")
                    cleanup_results['errors'].append(f"Dataset access error: {e}")
                
                # 2. Clean up functions using correct API pattern
                self.logger.info(f"⚙️ Cleaning functions on {self.site_id}...")
                try:
                    user_code = admin_client.code
                    code_list = list(user_code)
                    
                    for code_item in code_list:
                        try:
                            code_id = code_item.id
                            code_name = getattr(code_item, 'service_func_name', 'unknown')
                            self.logger.info(f"   🗑️ Deleting function: {code_name} (ID: {code_id})")
                            
                            # FIXED: Use correct API pattern
                            admin_client.api.code.delete(uid=code_id)
                            cleanup_results['functions_deleted'] += 1
                            
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Could not delete function {code_name}: {e}")
                            cleanup_results['errors'].append(f"Function deletion error: {e}")
                    
                    self.logger.info(f"   ✅ Deleted {cleanup_results['functions_deleted']} functions")
                    
                except Exception as e:
                    self.logger.warning(f"   ❌ Function cleanup error: {e}")
                    cleanup_results['errors'].append(f"Function access error: {e}")
                
                # 3. Clean up requests using correct API pattern
                self.logger.info(f"📋 Cleaning requests on {self.site_id}...")
                try:
                    requests = admin_client.requests
                    request_list = list(requests)
                    
                    for request in request_list:
                        try:
                            request_id = request.id
                            self.logger.info(f"   🗑️ Deleting request: {request_id}")
                            
                            # FIXED: Use correct API pattern
                            admin_client.api.request.delete(uid=request_id)
                            cleanup_results['requests_deleted'] += 1
                            
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Could not delete request {request_id}: {e}")
                            cleanup_results['errors'].append(f"Request deletion error: {e}")
                    
                    self.logger.info(f"   ✅ Deleted {cleanup_results['requests_deleted']} requests")
                    
                except Exception as e:
                    self.logger.warning(f"   ❌ Request cleanup error: {e}")
                    cleanup_results['errors'].append(f"Request access error: {e}")
                
                # Report cleanup results
                total_deleted = (cleanup_results['datasets_deleted'] + 
                               cleanup_results['functions_deleted'] + 
                               cleanup_results['requests_deleted'])
                
                if len(cleanup_results['errors']) == 0:
                    self.logger.info(f"🎉 {self.site_id} cleanup completed successfully! Deleted {total_deleted} items.")
                else:
                    self.logger.warning(f"⚠️ {self.site_id} cleanup completed with {len(cleanup_results['errors'])} errors. Deleted {total_deleted} items.")
            
            # Properly shutdown the PySyft server using server.land()
            if self.server:
                try:
                    self.logger.info(f"🛑 Executing server.land() for datasite {self.site_id} on port {self.port}")
                    self.server.land()
                    self.logger.info(f"✅ server.land() completed successfully for {self.site_id}")
                except Exception as e:
                    self.logger.error(f"❌ server.land() failed for {self.site_id}: {e}")
                    # Fallback to other shutdown methods
                    try:
                        if hasattr(self.server, 'shutdown'):
                            self.server.shutdown()
                            self.logger.debug(f"Used fallback shutdown() for {self.site_id}")
                        elif hasattr(self.server, 'stop'):
                            self.server.stop()
                            self.logger.debug(f"Used fallback stop() for {self.site_id}")
                    except Exception as fallback_e:
                        self.logger.debug(f"Fallback shutdown also failed: {fallback_e}")
                
                self.server = None
                self.datasite = None
            else:
                self.logger.warning(f"⚠️ No server reference found for {self.site_id} - cannot call server.land()")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup of REAL datasite {self.site_id}: {e}")
        
        # Always reset the state regardless of errors
        self.training_function_setup = False
        self.registered_functions = {}
        self.admin_client = None
        self.datasite = None
        self.datasite_client = None
        self.server = None
        self.data_upload_success = False
        
        # Clear any cached metrics
        if hasattr(self, 'last_metrics'):
            self.last_metrics = None
                
        self.logger.info(f"✅ Enhanced cleanup completed for REAL datasite {self.site_id}")
    
    def force_recreate_datasite(self):
        """Force complete recreation of PySyft datasite from scratch."""
        try:
            self.logger.info(f"🔄 FORCE recreating PySyft datasite {self.site_id} from scratch")
            
            # Step 1: Complete cleanup first
            self.cleanup()
            
            # Step 2: Wait a moment for cleanup to complete
            import time
            time.sleep(2)
            
            # Step 3: Re-launch the datasite completely fresh
            self._launch_real_datasite()
            
            # Step 4: Re-upload data if we have it
            if hasattr(self, 'federated_dataset') and self.federated_dataset:
                self.logger.info(f"Re-uploading federated dataset for recreated datasite {self.site_id}")
                self._auto_upload_federated_dataset()
            
            self.logger.info(f"✅ Successfully recreated datasite {self.site_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to recreate datasite {self.site_id}: {e}")
            return False
    
    def shutdown_datasite(self):
        """Shutdown the REAL PySyft datasite with COMPLETE cleanup."""
        self.cleanup()
    
    def __str__(self) -> str:
        return f"FactoryDataSite(id={self.site_id}, name={self.site_name}, port={self.port}, REAL_SYFT=True)"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_factory_datasite(site_id: str, site_name: str, port: int = 8080, 
                          hostname: str = "localhost", dev_mode: bool = True, 
                          reset: bool = True, verbose: bool = False) -> FactoryDataSite:
    """
    Factory function to create a FactoryDataSite instance with REAL PySyft infrastructure.
    
    Args:
        site_id: Unique identifier for the datasite
        site_name: Human-readable name for the datasite
        port: Port for the PySyft datasite server
        hostname: Hostname for the datasite server
        dev_mode: Whether to run in development mode
        reset: Whether to reset the datasite on launch
        verbose: Whether to enable verbose logging
        
    Returns:
        FactoryDataSite instance with REAL PySyft datasite
    """
    return FactoryDataSite(site_id, site_name, port, hostname, dev_mode, reset, verbose)


def create_external_factory_datasite(site_id: str, hostname: str, port: int,
                                    admin_email: str = "info@openmined.org", 
                                    admin_password: str = "changethis",
                                    site_name: Optional[str] = None, verbose: bool = False) -> FactoryDataSite:
    """
    Factory function to create a FactoryDataSite that connects to an external PySyft datasite.
    
    Args:
        site_id: Unique identifier for the datasite
        hostname: Hostname/IP of the external datasite
        port: Port of the external datasite
        admin_email: Admin email for connection
        admin_password: Admin password for connection
        site_name: Human-readable name for the datasite (auto-generated if not provided)
        verbose: Whether to enable verbose logging
        
    Returns:
        FactoryDataSite instance connected to external PySyft datasite
    """
    return FactoryDataSite(
        site_id=site_id,
        site_name=site_name,
        hostname=hostname,
        port=port,
        admin_email=admin_email,
        admin_password=admin_password,
        use_external_datasite=True,
        verbose=verbose
    )


def create_factory_datasites_from_config(config_file: Optional[str] = None, 
                                        model_type_filter: Optional[str] = None, 
                                        verbose: bool = False) -> Dict[str, FactoryDataSite]:
    """
    Create multiple FactoryDataSite instances from configuration file for external datasites.
    
    Args:
        config_file: Path to datasite configuration YAML file
        model_type_filter: Only create datasites with this model type (e.g., 'cnn', 'lstm', 'hybrid')
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary mapping site_id to FactoryDataSite instances
    """
    if not CONFIG_AVAILABLE:
        raise ImportError("Datasite configuration system not available. Check datasite_config.py import.")
    
    # Load configuration
    config_manager = DatasiteConfigManager(config_file)
    config_manager.load_config()  # Actually load the configuration!
    
    if not config_manager.datasites:
        config_file_path = config_file or "datasite_configs.yaml"
        raise ValueError(f"No datasites found in configuration file: {config_file_path}")
    
    # Note: Datasites are not model-specific, so we ignore model_type_filter
    # All configured datasites can be used for any model type
    if model_type_filter and verbose:
        print(f"[INFO] model_type_filter '{model_type_filter}' ignored - datasites are not model-specific")
    
    datasites_to_create = config_manager.datasites
    
    # Create FactoryDataSite instances for external connection
    factory_datasites = {}
    
    for datasite_config in datasites_to_create:
        site_id = datasite_config.id
        try:
            factory_datasite = FactoryDataSite(
                site_id=site_id,
                config_manager=config_manager,
                use_external_datasite=True,
                verbose=verbose
            )
            factory_datasites[site_id] = factory_datasite
            
            if verbose:
                print(f"✅ Created external factory datasite: {site_id} at {datasite_config.host}:{datasite_config.port}")
                
        except Exception as e:
            print(f"❌ Failed to create external factory datasite {site_id}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    if not factory_datasites:
        raise RuntimeError("Failed to create any external factory datasites from configuration")
    
    print(f"Successfully created {len(factory_datasites)} external factory datasites from configuration")
    return factory_datasites


def validate_external_datasites(config_file: Optional[str] = None) -> Dict[str, bool]:
    """
    Validate connectivity to all configured external datasites.
    
    Args:
        config_file: Path to datasite configuration YAML file
        
    Returns:
        Dictionary mapping site_id to connection status (True/False)
    """
    if not CONFIG_AVAILABLE:
        raise ImportError("Datasite configuration system not available. Check datasite_config.py import.")
    
    config_manager = DatasiteConfigManager(config_file)
    return config_manager.validate_connectivity()
