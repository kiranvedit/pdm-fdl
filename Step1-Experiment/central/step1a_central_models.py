#!/usr/bin/env python3
"""
Step 3A: Central Models Implementation with Statistical Evaluation
Creates and trains CNN, LSTM, and CNN-LSTM models centrally with comprehensive statistical comparison
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Add shared directory to path for imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, shared_path)
sys.path.insert(0, os.path.join(shared_path, 'utils'))
sys.path.insert(0, os.path.join(shared_path, 'models'))

# Import using importlib to ensure proper loading
import importlib.util

# Load pdm_models
pdm_models_path = os.path.join(shared_path, 'models', 'pdm_models.py')
spec = importlib.util.spec_from_file_location("pdm_models", pdm_models_path)
pdm_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdm_models)
CentralCNNModel = pdm_models.CentralCNNModel
CentralLSTMModel = pdm_models.CentralLSTMModel
CentralHybridModel = pdm_models.CentralHybridModel

# Load step1_data_utils
data_utils_path = os.path.join(shared_path, 'utils', 'step1_data_utils.py')
spec = importlib.util.spec_from_file_location("step1_data_utils", data_utils_path)
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)
ProcessedDataLoader = data_utils.DataLoader

# Load dashboard tracker
dashboard_path = os.path.join(shared_path, 'utils', 'dashboard_tracker.py')
spec = importlib.util.spec_from_file_location("dashboard_tracker", dashboard_path)
dashboard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard_module)
ExperimentDashboard = dashboard_module.ExperimentDashboard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
epochs = 50  # Default epochs for training

class CentralModelTrainer:
    """Central Model Training and Hyperparameter Tuning"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.training_history = {}
        self.best_models = {}
        self.hyperparameter_results = {}

        # Initialize dashboard tracker
        from dashboard_tracker import create_dashboard
        self.dashboard = create_dashboard()

        print(f"🖥️ Central Model Trainer initialized on {device}")
        print(f"📊 Dashboard tracking enabled: {self.dashboard.get_dashboard_url()}")
    
    def load_processed_data(self, data_dir="/workspaces/pdm-fdl/shared/processed_data"):
        """Load preprocessed data from Step 1"""
        print(f"📂 Loading preprocessed data from: {data_dir}")
        
        # Load processed data using Step 1 data loader
        data_loader = ProcessedDataLoader(data_dir)
        
        # Load tabular data (main training data)
        tabular_data = data_loader.load_tabular_data()
        
        X_train = tabular_data['X_train']
        y_train = tabular_data['y_train'].ravel()  # Flatten to 1D
        X_val = tabular_data['X_val'] 
        y_val = tabular_data['y_val'].ravel()
        X_test = tabular_data['X_test']
        y_test = tabular_data['y_test'].ravel()
        
        # Load metadata for feature information
        metadata = data_loader.metadata
        feature_names = metadata['feature_names']
        
        print(f"📊 Data loaded successfully:")
        print(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        print(f"   Features: {feature_names}")
        print(f"   Classes: {len(np.unique(y_train))} unique classes")
        
        # Log data info to dashboard
        data_info = {
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(y_train)),
            'feature_names': feature_names
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test, data_info
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Create PyTorch data loaders"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader, model_name,
                   epochs=epochs, learning_rate=0.001, patience=10):
        """Train a single model with early stopping and dashboard tracking"""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'training_time': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            history['training_time'].append(epoch_time)
            # Log to dashboard
            metrics = {
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc,
                'learning_rate': current_lr,
                'training_time': epoch_time
            }
            self.dashboard.log_epoch_metrics(epoch, metrics)
            try:
                self.dashboard.save_dashboard("step1a_central_models_perf_dashboard.html")
            except Exception as e:
                print(f"⚠️ Could not save dashboard after epoch {epoch}: {e}")
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            # Only print epoch info for CNN, or for LSTM/Hybrid if not called from hyperparameter_tuning
            if (model_name.endswith('_final') or model_name.endswith('_run_1') or model_name.endswith('_run_2') or model_name.endswith('_run_3') or model_name.endswith('_run_4') or model_name.endswith('_run_5') or model_name.startswith('CNN')):
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch:3d}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, Time {epoch_time:.2f}s")
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - total_start_time
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        history['total_training_time'] = total_time
        history['best_val_accuracy'] = best_val_acc
        
        print(f"   ✅ Training completed in {total_time:.2f}s")
        print(f"   🎯 Best validation accuracy: {best_val_acc:.4f}")
        # Save dashboard at the end of training
        try:
            self.dashboard.save_dashboard("step1a_central_models_perf_dashboard.html")
        except Exception as e:
            print(f"⚠️ Could not save dashboard after training: {e}")
        return model, history
    
    def hyperparameter_tuning(self, model_class, model_name, 
                            X_train, y_train, X_val, y_val, 
                            param_grid, max_trials=10):
        """Perform hyperparameter tuning with dashboard tracking"""
        print(f"\n🔧 Hyperparameter tuning for {model_name}")
        print(f"   Parameters to tune: {list(param_grid.keys())}")
        
        # Log experiment start to dashboard
        data_info = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'num_features': X_train.shape[1] if len(X_train.shape) > 1 else 1
        }
        
        best_score = 0.0
        best_params = None
        best_model = None
        best_history = None
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Limit trials if too many combinations
        if len(param_combinations) > max_trials:
            param_combinations = np.random.choice(
                param_combinations, max_trials, replace=False
            ).tolist()
        
        print(f"   Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for i, params in enumerate(param_combinations):
            print(f"   Trial {i+1}/{len(param_combinations)}: {params}")
            
            # Log hyperparameter trial to dashboard
            self.dashboard.log_experiment_start(f"{model_name}_trial_{i+1}", params, data_info)
            
            try:
                # Extract model-specific parameters only
                model_params = {}
                training_params = {'learning_rate', 'batch_size'}
                for key, value in params.items():
                    if key not in training_params:
                        model_params[key] = value
                # Always set input_dim and num_classes for LSTM/Hybrid/CNN
                if model_name == 'LSTM' or model_name == 'Hybrid':
                    model_params['input_dim'] = X_train.shape[1]
                    model_params['num_classes'] = len(np.unique(y_train))
                elif model_name == 'CNN':
                    model_params['input_dim'] = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
                    model_params['num_classes'] = len(np.unique(y_train))
                model = model_class(**model_params)
                
                # Create data loaders
                train_loader, val_loader = self.create_data_loaders(
                    X_train, y_train, X_val, y_val, 
                    batch_size=params.get('batch_size', 32)
                )
                
                # Train model
                trained_model, history = self.train_model(
                    model, train_loader, val_loader, f"{model_name}_trial_{i+1}",
                    epochs=epochs,  # Reduced for hyperparameter tuning
                    learning_rate=params.get('learning_rate', 0.001)
                )
                
                val_score = history['best_val_accuracy']
                
                # Store results
                result_metrics = {
                    'val_accuracy': val_score,
                    'training_time': history['total_training_time'],
                    'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
                    'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0
                }
                
                results.append({
                    'params': params,
                    'val_accuracy': val_score,
                    'training_time': history['total_training_time']
                })
                
                # Log to dashboard
                self.dashboard.log_experiment_completion(f"{model_name}_trial_{i+1}", result_metrics, params)
                # Save dashboard after each trial
                try:
                    self.dashboard.save_dashboard("step1a_central_models_perf_dashboard.html")
                except Exception as e:
                    print(f"⚠️ Could not save dashboard after trial {i+1}: {e}")
                # Update best model
                if val_score > best_score:
                    best_score = val_score
                    best_params = params
                    best_model = trained_model
                    best_history = history
                print(f"     Validation accuracy: {val_score:.4f}")
            except Exception as e:
                print(f"     Failed: {e}")
                continue
        
        self.hyperparameter_results[model_name] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
        
        print(f"   🏆 Best parameters: {best_params}")
        print(f"   🎯 Best validation accuracy: {best_score:.4f}")
        # Save dashboard after hyperparameter tuning
        try:
            self.dashboard.save_dashboard("step1a_central_models_perf_dashboard.html")
        except Exception as e:
            print(f"⚠️ Could not save dashboard after hyperparameter tuning: {e}")
        return best_model, best_history, best_params
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model on test set with comprehensive metrics"""
        model.eval()
        model = model.to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
            # Get probabilities for AUC calculation
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Extract precision, recall, f1_score from weighted average
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Calculate AUC for multi-class
        try:
            auc_score = roc_auc_score(y_test, probabilities, multi_class='ovr', average='weighted')
        except:
            auc_score = 0.0
        
        # Detailed metrics
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # Per-sample inference time
        per_sample_time = inference_time / len(X_test) * 1000  # milliseconds
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'inference_time_total': inference_time,
            'inference_time_per_sample': per_sample_time,
            'predictions': predictions.tolist()
        }
        
        print(f"\n📊 {model_name} Test Results:")
        print(f"   🎯 Accuracy: {accuracy:.4f}")
        print(f"   🔍 Precision: {precision:.4f}")
        print(f"   📊 Recall: {recall:.4f}")
        print(f"   ⚖️  F1-Score: {f1:.4f}")
        print(f"   📈 AUC: {auc_score:.4f}")
        print(f"   ⏱️  Total inference time: {inference_time:.4f}s")
        print(f"   ⚡ Per-sample inference: {per_sample_time:.2f}ms")
        
        return results
    
    def save_results(self, results, filepath="central_models_results.json"):
        """Save all results to file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filepath}")
    
    def run_multiple_evaluations(self, model_class, model_params, X_train, y_train, 
                                X_test, y_test, model_name, num_runs=5, 
                                learning_rate=0.001, batch_size=32):
        """Run multiple training and evaluation cycles for statistical analysis"""
        print(f"   Running {num_runs} independent training cycles...")
        
        results = {
            'training_times': [],
            'training_accuracies': [],
            'test_accuracies': [],
            'inference_times': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'auc_scores': []
        }
        
        for run in range(num_runs):
            print(f"   Run {run + 1}/{num_runs}...")
            
            # Create fresh model for each run
            model = model_class(**model_params)
            
            # Create data loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Dummy validation loader for training function
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                batch_size=batch_size, shuffle=False
            )
            
            # Train model
            trained_model, history = self.train_model(
                model, train_loader, val_loader, f"{model_name}_run_{run+1}",
                epochs=epochs,  # Fewer epochs for multiple runs
                learning_rate=learning_rate,
                patience=10
            )
            
            # Evaluate model
            test_results = self.evaluate_model(trained_model, X_test, y_test, f"{model_name}_run_{run+1}")
            
            # Store results
            results['training_times'].append(history['total_training_time'])
            results['training_accuracies'].append(history['best_val_accuracy'])
            results['test_accuracies'].append(test_results['accuracy'])
            results['inference_times'].append(test_results['inference_time_per_sample'])
            results['precisions'].append(test_results['precision'])
            results['recalls'].append(test_results['recall'])
            results['f1_scores'].append(test_results['f1_score'])
            results['auc_scores'].append(test_results['auc_score'])
        
        # Calculate statistics
        stats = {}
        for metric, values in results.items():
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        print(f"   ✅ Completed {num_runs} runs")
        print(f"   📊 Mean test accuracy: {stats['test_accuracies']['mean']:.4f} ± {stats['test_accuracies']['std']:.4f}")
        print(f"   🔍 Mean precision: {stats['precisions']['mean']:.4f} ± {stats['precisions']['std']:.4f}")
        print(f"   📊 Mean recall: {stats['recalls']['mean']:.4f} ± {stats['recalls']['std']:.4f}")
        print(f"   ⚖️  Mean F1-score: {stats['f1_scores']['mean']:.4f} ± {stats['f1_scores']['std']:.4f}")
        print(f"   📈 Mean AUC: {stats['auc_scores']['mean']:.4f} ± {stats['auc_scores']['std']:.4f}")
        
        return {
            'raw_results': results,
            'statistics': stats,
            'num_runs': num_runs
        }

def main():
    """Main function to run central model training and evaluation"""
    print("=" * 80)
    print("🏭 STEP 3A: CENTRAL MODELS TRAINING")
    print("   Industrial Predictive Maintenance Models")
    print("=" * 80)
    
    # Initialize trainer
    trainer = CentralModelTrainer()
    
    # Load preprocessed data from Step 1
    X_train, X_val, X_test, y_train, y_val, y_test, data_info = trainer.load_processed_data()
    
    # Define hyperparameter grids (adapted for 10 features from Step 1)
    hyperparameter_grids = {
        'CNN': {
            'conv_filters': [[32, 64], [32, 64, 128], [64, 128, 256]],
            'fc_hidden': [[64], [128, 64], [256, 128]],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32]
        },
        'LSTM': {
            'hidden_dim': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout_rate': [0.2, 0.3, 0.4],
            'bidirectional': [True, False],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32]
        },
        'Hybrid': {
            'cnn_filters': [[32, 64], [64, 128]],
            'lstm_hidden': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32]
        }
    }
    
    # Model classes (updated for 10 features)
    model_classes = {
        'CNN': CentralCNNModel,
        'LSTM': CentralLSTMModel,
        'Hybrid': CentralHybridModel
    }
    
    # Train and evaluate all models
    all_results = {}
    
    for model_name in ['CNN', 'LSTM', 'Hybrid']:
        print(f"\n{'='*60}")
        print(f"🏗️ TRAINING {model_name} MODEL")
        print(f"{'='*60}")
        
        # Adjust input dimensions based on model type
        if model_name == 'CNN':
            # Reshape for CNN (add channel dimension)
            X_train_model = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val_model = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            X_test_model = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        else:
            X_train_model = X_train
            X_val_model = X_val
            X_test_model = X_test
        
        # Log experiment start to dashboard
        trainer.dashboard.log_experiment_start(model_name, {}, data_info)
        
        # Hyperparameter tuning
        best_model, best_history, best_params = trainer.hyperparameter_tuning(
            model_classes[model_name], model_name,
            X_train_model, y_train, X_val_model, y_val,
            hyperparameter_grids[model_name],
            max_trials=8
        )
        
        # Check if hyperparameter tuning was successful
        if best_params is None:
            print(f"⚠️ Hyperparameter tuning failed for {model_name}, using default parameters")
            # Use default parameters
            if model_name == 'CNN':
                best_params = {
                    'input_dim': 10, 'conv_filters': [32, 64], 'fc_hidden': [128], 
                    'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 32
                }
            elif model_name == 'LSTM':
                best_params = {
                    'input_dim': 10, 'hidden_dim': 64, 'num_layers': 2, 'dropout_rate': 0.3,
                    'bidirectional': True, 'learning_rate': 0.001, 'batch_size': 32
                }
            elif model_name == 'Hybrid':
                best_params = {
                    'input_dim': 10, 'cnn_filters': [32, 64], 'lstm_hidden': 64, 'dropout_rate': 0.3,
                    'learning_rate': 0.001, 'batch_size': 32
                }
        
        # Final training with best parameters
        print(f"\n🔧 Final training with best parameters...")
        
        # Extract model-specific parameters only
        model_params = {}
        training_params = {'learning_rate', 'batch_size'}
        for key, value in best_params.items():
            if key not in training_params:
                model_params[key] = value

        # Set input dimension for all models
        # For LSTM/Hybrid, check model signature for input_size/input_dim
        if model_name == 'LSTM':
            model_params['input_dim'] = X_train_model.shape[1]
            model_params['num_classes'] = len(np.unique(y_train))
        elif model_name == 'Hybrid':
            model_params['input_dim'] = X_train_model.shape[1]
            model_params['num_classes'] = len(np.unique(y_train))
        else:
            model_params['input_dim'] = X_train_model.shape[2] if len(X_train_model.shape) == 3 else X_train_model.shape[1]
            model_params['num_classes'] = len(np.unique(y_train))

        final_model = model_classes[model_name](**model_params)
        train_loader_final, val_loader_final = trainer.create_data_loaders(
            X_train_model, y_train, X_val_model, y_val,
            batch_size=best_params.get('batch_size', 32)
        )
        
        final_model, final_history = trainer.train_model(
            final_model, train_loader_final, val_loader_final, f"{model_name}_final",
            epochs=epochs, learning_rate=best_params.get('learning_rate', 0.001),
            patience=15
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate_model(final_model, X_test_model, y_test, model_name)
        
        # Multiple runs for statistical analysis
        print(f"\n🔢 Running multiple evaluations for statistical analysis...")
        multiple_results = trainer.run_multiple_evaluations(
            model_classes[model_name], model_params, X_train_model, y_train, 
            X_test_model, y_test, model_name, num_runs=5,
            learning_rate=best_params.get('learning_rate', 0.001),
            batch_size=best_params.get('batch_size', 32)
        )
        
        # Final metrics for dashboard
        final_metrics = {
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_f1_score': test_results['f1_score'],
            'training_time': final_history['total_training_time'],
            'inference_time': test_results['inference_time_per_sample']
        }
        
        # Log experiment completion to dashboard
        trainer.dashboard.log_experiment_completion(model_name, final_metrics, best_params)
        
        # Store results
        all_results[model_name] = {
            'best_hyperparameters': best_params,
            'hyperparameter_tuning': trainer.hyperparameter_results[model_name],
            'training_history': final_history,
            'test_results': test_results,
            'multiple_runs_results': multiple_results
        }
        
        # Save model
        model_path = f"models/central_{model_name.lower()}_model.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(final_model.state_dict(), model_path)
        print(f"💾 Model saved: {model_path}")
    
    # Save comprehensive results
    trainer.save_results(all_results, "results/central_models_results.json")
    
    # Model comparison for dashboard
    model_metrics = []
    for model_name, results in all_results.items():
        model_metrics.append({
            'model': model_name,
            'accuracy': results['test_results']['accuracy'],
            'precision': results['test_results']['precision'],
            'recall': results['test_results']['recall'],
            'f1_score': results['test_results']['f1_score'],
            'training_time': results['training_history']['total_training_time'],
            'inference_time': results['test_results']['inference_time_per_sample']
        })
    
    trainer.dashboard.add_model_comparison({
        'model_metrics': model_metrics,
        'experiment_name': 'central_model_comparison'
    })
    
    # Performance comparison
    print(f"\n{'='*80}")
    print("📊 CENTRAL MODELS PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        test_acc = results['test_results']['accuracy']
        train_time = results['training_history']['total_training_time']
        inference_time = results['test_results']['inference_time_per_sample']
        
        print(f"\n🏗️ {model_name} Model:")
        print(f"   🎯 Test Accuracy: {test_acc:.4f}")
        print(f"   ⏱️  Training Time: {train_time:.2f}s")
        print(f"   ⚡ Inference Time: {inference_time:.2f}ms/sample")
    
    # Find best model
    best_model = max(all_results.items(), 
                    key=lambda x: x[1]['test_results']['accuracy'])
    
    print(f"\n🏆 Best Central Model: {best_model[0]}")
    print(f"   🎯 Accuracy: {best_model[1]['test_results']['accuracy']:.4f}")
    
    print(f"\n✅ Central models training completed!")
    print(f"📁 Results saved in: results/")
    print(f"🏗️ Models saved in: models/")
    # Always save dashboard at the end
    try:
        trainer.dashboard.save_dashboard("step1a_central_models_perf_dashboard.html")
        dashboard_path = trainer.dashboard.get_dashboard_url()
        print(f"📊 Dashboard available at: {dashboard_path}")
    except Exception as e:
        print(f"⚠️ Could not save dashboard at end: {e}")
    return all_results

class StatisticalEvaluator:
    """Enhanced evaluator for comprehensive statistical comparison"""
    
    def __init__(self, output_dir="results/statistical_comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.results = {
            'centralized': {
                'training_time': [],
                'training_accuracy': [],
                'test_accuracy': [],
                'inference_time': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'auc_score': []
            },
            'federated': {
                'training_time': [],
                'training_accuracy': [],
                'test_accuracy': [],
                'inference_time': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'auc_score': []
            }
        }
    
    def measure_inference_time(self, model, sample_input, num_runs=100):
        """Measure average inference time per sample"""
        model.eval()
        times = []
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Measure
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(sample_input)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return np.mean(times)
    
    def comprehensive_model_evaluation(self, model, test_loader):
        """Comprehensive model evaluation with all metrics"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate all metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # AUC for multi-class
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc
        }
    
    def train_and_evaluate_centralized(self, model_class, train_loader, test_loader, 
                                     num_runs=5, epochs=epochs, **model_kwargs):
        """Train and evaluate centralized model multiple times"""
        logger.info(f"Training centralized {model_class.__name__} for {num_runs} runs")
        
        for run in range(num_runs):
            logger.info(f"Centralized run {run + 1}/{num_runs}")
            
            # Create fresh model
            model = model_class(**model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Training
            start_time = time.time()
            model.train()
            final_train_acc = 0
            
            for epoch in range(epochs):
                correct = 0
                total = 0
                
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
                
                final_train_acc = correct / total
            
            training_time = time.time() - start_time
            
            # Test evaluation
            test_metrics = self.comprehensive_model_evaluation(model, test_loader)
            
            # Inference time measurement
            sample_input = next(iter(test_loader))[0][:1].to(self.device)
            inference_time = self.measure_inference_time(model, sample_input)
            
            # Store results
            self.results['centralized']['training_time'].append(training_time)
            self.results['centralized']['training_accuracy'].append(final_train_acc)
            self.results['centralized']['test_accuracy'].append(test_metrics['accuracy'])
            self.results['centralized']['inference_time'].append(inference_time)
            self.results['centralized']['precision'].append(test_metrics['precision'])
            self.results['centralized']['recall'].append(test_metrics['recall'])
            self.results['centralized']['f1_score'].append(test_metrics['f1_score'])
            self.results['centralized']['auc_score'].append(test_metrics['auc_score'])
            
            logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    def perform_statistical_tests(self):
        """Perform statistical tests between centralized and federated results"""
        logger.info("Performing statistical tests")
        
        statistical_results = {}
        
        for metric in self.results['centralized'].keys():
            cent_values = self.results['centralized'][metric]
            fed_values = self.results['federated'][metric]
            
            if len(cent_values) > 1 and len(fed_values) > 1:
                # t-test
                t_stat, p_value = stats.ttest_ind(cent_values, fed_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(cent_values) - 1) * np.var(cent_values, ddof=1) + 
                                    (len(fed_values) - 1) * np.var(fed_values, ddof=1)) / 
                                   (len(cent_values) + len(fed_values) - 2))
                
                if pooled_std > 0:
                    cohens_d = (np.mean(cent_values) - np.mean(fed_values)) / pooled_std
                else:
                    cohens_d = 0.0
                
                statistical_results[metric] = {
                    'centralized_mean': np.mean(cent_values),
                    'centralized_std': np.std(cent_values),
                    'federated_mean': np.mean(fed_values),
                    'federated_std': np.std(fed_values),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                }
        
        return statistical_results
    
    def create_comparison_visualizations(self, statistical_results):
        """Create comprehensive comparison visualizations"""
        logger.info("Creating comparison visualizations")
        
        # Box plot comparison
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle('Statistical Comparison: Centralized vs Federated Models', 
                    fontsize=16, fontweight='bold')
        
        metrics = ['training_time', 'training_accuracy', 'test_accuracy', 'inference_time',
                  'precision', 'recall', 'f1_score', 'auc_score']
        
        for idx, metric in enumerate(metrics):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]
            
            data = [self.results['centralized'][metric], self.results['federated'][metric]]
            bp = ax.boxplot(data, labels=['Centralized', 'Federated'], patch_artist=True)
            
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            # Statistical significance indicator
            if metric in statistical_results and statistical_results[metric]['significant']:
                ax.text(0.5, 0.95, 'p < 0.05*', transform=ax.transAxes, 
                       ha='center', va='top', fontweight='bold', color='red')
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/comparison_boxplots.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical summary table
        self.create_statistical_table(statistical_results)
    
    def create_statistical_table(self, statistical_results):
        """Create statistical summary table"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        table_data = []
        for metric, stats in statistical_results.items():
            table_data.append([
                metric.replace('_', ' ').title(),
                f"{stats['centralized_mean']:.4f} ± {stats['centralized_std']:.4f}",
                f"{stats['federated_mean']:.4f} ± {stats['federated_std']:.4f}",
                f"{stats['t_statistic']:.3f}",
                f"{stats['p_value']:.4f}",
                f"{stats['cohens_d']:.3f}",
                "Yes*" if stats['significant'] else "No"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Centralized\n(Mean ± Std)', 'Federated\n(Mean ± Std)', 
                                 't-statistic', 'p-value', "Cohen's d", 'Significant'],
                        cellLoc='center', loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.axis('off')
        ax.set_title('Statistical Comparison Results: Centralized vs Federated Models', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(f"{self.output_dir}/visualizations/statistical_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_statistical_results(self, statistical_results):
        """Save statistical results and generate report"""
        logger.info("Saving statistical results")
        
        # Save JSON results
        with open(f"{self.output_dir}/statistical_results.json", 'w') as f:
            json_results = {}
            for metric, stats in statistical_results.items():
                json_results[metric] = {
                    k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                    for k, v in stats.items()
                }
            json.dump(json_results, f, indent=2)
        
        # Save raw results
        with open(f"{self.output_dir}/raw_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate report
        report = []
        report.append("# Comprehensive Statistical Comparison Report")
        report.append("=" * 50)
        report.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of runs per model: {len(self.results['centralized']['test_accuracy'])}")
        report.append("")
        
        significant_metrics = [m for m, s in statistical_results.items() if s['significant']]
        if significant_metrics:
            report.append(f"Statistically significant differences found in: {', '.join(significant_metrics)}")
        else:
            report.append("No statistically significant differences found.")
        
        report.append("\n## Detailed Results")
        report.append("-" * 30)
        
        for metric, stats in statistical_results.items():
            report.append(f"\n### {metric.replace('_', ' ').title()}")
            report.append(f"- Centralized: {stats['centralized_mean']:.4f} ± {stats['centralized_std']:.4f}")
            report.append(f"- Federated: {stats['federated_mean']:.4f} ± {stats['federated_std']:.4f}")
            report.append(f"- t-statistic: {stats['t_statistic']:.3f}")
            report.append(f"- p-value: {stats['p_value']:.4f}")
            report.append(f"- Cohen's d: {stats['cohens_d']:.3f}")
            report.append(f"- Significant: {'Yes' if stats['significant'] else 'No'}")
        
        with open(f"{self.output_dir}/statistical_report.md", 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Statistical results saved to: {self.output_dir}")

def run_statistical_comparison(model_type='cnn', num_runs=5, epochs=epochs):
    """Run comprehensive statistical comparison for a model type"""
    logger.info(f"Starting statistical comparison for {model_type.upper()} model")
    
    # Load data
    logger.info("Loading data...")
    data_dir = "shared/data"
    
    try:
        X_train = np.load(f"{data_dir}/X_train.npy")
        X_test = np.load(f"{data_dir}/X_test.npy")
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        
        logger.info(f"Data loaded: Train {X_train.shape}, Test {X_test.shape}")
        
        # Handle dimension mismatch for 1D CNN
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create evaluator
        evaluator = StatisticalEvaluator(f"results/statistical_comparison_{model_type}")
        
        # Model selection
        model_classes = {
            'cnn': CentralCNNModel,
            'lstm': CentralLSTMModel, 
            'hybrid': CentralHybridModel
        }
        
        model_class = model_classes.get(model_type, CentralCNNModel)
        model_kwargs = {'input_dim': 5 if model_type == 'lstm' else X_train.shape[-1], 'num_classes': 6}
        
        # Train centralized models
        evaluator.train_and_evaluate_centralized(
            model_class, train_loader, test_loader, 
            num_runs=num_runs, epochs=epochs, **model_kwargs
        )
        
        logger.info("Centralized evaluation completed")
        return evaluator
        
    except Exception as e:
        logger.error(f"Error in statistical comparison: {e}")
        raise
    
    return all_results

if __name__ == "__main__":
    main()
