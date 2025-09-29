"""
Federated Learning Server Implementation
Supports FedAvg, FedDyn, FedProx, FedNova algorithms with comprehensive metrics tracking
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Any
from experiment_logger import get_experiment_logger


class FederatedServer:
    """
    Federated learning server supporting all four algorithms with comprehensive metrics tracking:
    - FedAvg: Standard federated averaging
    - FedDyn: Dynamic regularization for improved convergence
    - FedProx: Proximal regularization for system heterogeneity
    - FedNova: Variance reduction for non-IID data
    
    Enhanced Features:
    - Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUC
    - Training time tracking: Round time, aggregation time
    - Early stopping support: Convergence detection
    """
    
    def __init__(self, model_class, model_params: Dict, aggregation_method: str = 'fedavg',
                 device: str = 'cpu', privacy_enabled: bool = True, early_stopping_patience: int = 10,
                 early_stopping_min_delta: float = 0.001):
        self.model_class = model_class
        self.model_params = model_params
        self.aggregation_method = aggregation_method
        self.device = device
        self.privacy_enabled = privacy_enabled
        
        # Initialize global model
        self.global_model = model_class(**model_params).to(device)
        
        # Algorithm-specific state
        self.algorithm_state = {
            'round': 0,
            'feddyn_h': None,  # FedDyn regularization parameter
            'fednova_momentum': None,  # FedNova momentum buffer
            'client_momentums': {}  # Client-specific momentum for FedNova
        }
        
        # Enhanced metrics tracking with all required fields
        self.training_history = []
        self.round_metrics = []
        self.aggregation_times = []
        self.round_times = []
        self.client_metrics_history = []
        
        # Early stopping configuration
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False
        
        # Best performance tracking
        self.best_accuracy = 0.0
        self.best_f1_score = 0.0
        self.best_precision = 0.0
        self.best_recall = 0.0
        self.best_auc = 0.0
        self.best_round = 0
        self.best_metrics = {}
        
        # Total training time tracking
        self.experiment_start_time = None
        self.total_training_time = 0.0
        
        print(f"Federated Server initialized")
        print(f"   Model: {model_class.__name__}")
        print(f"   Algorithm: {aggregation_method.upper()}")
        print(f"   Device: {device}")
        print(f"   Privacy: {'Enabled' if privacy_enabled else 'Disabled'}")
    
    def get_global_model_state(self) -> Dict:
        """Get current global model state for distribution to clients"""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}
    
    def aggregate_updates(self, client_updates: List[Dict], algorithm_kwargs: Dict = None) -> None:
        """
        Aggregate client updates using the specified algorithm with timing
        
        Args:
            client_updates: List of client update dictionaries containing model_update and metadata
            algorithm_kwargs: Algorithm-specific parameters
        """
        algorithm_kwargs = algorithm_kwargs or {}
        
        aggregation_start_time = time.time()
        round_start_time = time.time()
        self.algorithm_state['round'] += 1
        
        # Initialize experiment timer if first round
        if self.experiment_start_time is None:
            self.experiment_start_time = time.time()
        
        print(f"Server aggregating {len(client_updates)} updates (Round {self.algorithm_state['round']})")
        print(f"   Algorithm: {self.aggregation_method.upper()}")
        
        # Store client metrics for this round
        round_client_metrics = []
        for update in client_updates:
            client_summary = {
                'round': self.algorithm_state['round'],
                'client_id': update.get('client_id', 'unknown'),
                'accuracy': update.get('local_accuracy', 0.0),
                'precision': update.get('local_precision', 0.0),
                'recall': update.get('local_recall', 0.0),
                'f1_score': update.get('local_f1_score', 0.0),
                'auc': update.get('local_auc', 0.0),
                'training_time': update.get('training_time', 0.0),
                'inference_time': update.get('inference_time', 0.0),
                'sample_inference_time': update.get('sample_inference_time', 0.0),
                'num_samples': update.get('num_samples', 0)
            }
            round_client_metrics.append(client_summary)
        
        self.client_metrics_history.extend(round_client_metrics)
        
        # Dispatch to algorithm-specific aggregation
        if self.aggregation_method == 'fedavg':
            self._fedavg_aggregation(client_updates)
        elif self.aggregation_method == 'feddyn':
            self._feddyn_aggregation(client_updates, algorithm_kwargs)
        elif self.aggregation_method == 'fedprox':
            self._fedprox_aggregation(client_updates, algorithm_kwargs)
        elif self.aggregation_method == 'fednova':
            self._fednova_aggregation(client_updates, algorithm_kwargs)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        aggregation_time = time.time() - aggregation_start_time
        round_time = time.time() - round_start_time
        self.aggregation_times.append(aggregation_time)
        self.round_times.append(round_time)
        
        # Update total training time
        self.total_training_time = time.time() - self.experiment_start_time
        
        print(f"Aggregation complete (Aggregation: {aggregation_time:.2f}s, Round: {round_time:.2f}s)")
    
    def _fedavg_aggregation(self, client_updates: List[Dict]) -> None:
        """Standard FedAvg aggregation with weighted averaging"""
        print("   FedAvg: Weighted averaging")
        
        # Calculate total samples for weighted averaging
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Weighted averaging
        for name, param in self.global_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param)
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                if 'model_update' in update:
                    client_param = update['model_update'][name]
                else:
                    print(f"Warning: model_update missing for client {update.get('client_id', 'unknown')}")
                    continue
                aggregated_params[name] += weight * client_param
        
        # Update global model
        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                param.data.copy_(aggregated_params[name])
    
    def _feddyn_aggregation(self, client_updates: List[Dict], algorithm_kwargs: Dict) -> None:
        """FedDyn aggregation with dynamic regularization"""
        alpha = algorithm_kwargs.get('alpha', 0.01)
        print(f"   FedDyn: Dynamic regularization (alpha={alpha})")
        
        # Initialize h parameter if first round
        if self.algorithm_state['feddyn_h'] is None:
            self.algorithm_state['feddyn_h'] = {
                name: torch.zeros_like(param) 
                for name, param in self.global_model.named_parameters()
            }
        
        # Standard FedAvg aggregation first
        total_samples = sum(update['num_samples'] for update in client_updates)
        aggregated_params = {}
        
        for name, param in self.global_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param)
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                if 'model_update' in update:
                    client_param = update['model_update'][name]
                    aggregated_params[name] += weight * client_param
        
        # Apply FedDyn dynamic regularization
        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                # Update h parameter
                self.algorithm_state['feddyn_h'][name] += alpha * (aggregated_params[name] - param.data)
                
                # Update global model with regularization
                param.data.copy_(aggregated_params[name] - (1.0 / alpha) * self.algorithm_state['feddyn_h'][name])
    
    def _fedprox_aggregation(self, client_updates: List[Dict], algorithm_kwargs: Dict) -> None:
        """FedProx aggregation (same as FedAvg, proximal term applied during training)"""
        mu = algorithm_kwargs.get('mu', 0.01)
        print(f"   FedProx: Proximal regularization (mu={mu})")
        
        # FedProx uses standard FedAvg aggregation
        # The proximal term is applied during client training
        self._fedavg_aggregation(client_updates)
    
    def _fednova_aggregation(self, client_updates: List[Dict], algorithm_kwargs: Dict) -> None:
        """FedNova aggregation with variance reduction"""
        beta = algorithm_kwargs.get('beta', 0.9)
        print(f"   FedNova: Variance reduction (beta={beta})")
        
        # Calculate effective number of steps (tau_eff)
        total_steps = sum(update.get('local_steps', 1) for update in client_updates)
        num_clients = len(client_updates)
        tau_eff = total_steps / num_clients if num_clients > 0 else 1
        
        # Calculate weighted model differences
        total_samples = sum(update['num_samples'] for update in client_updates)
        aggregated_diff = {}
        
        for name, param in self.global_model.named_parameters():
            aggregated_diff[name] = torch.zeros_like(param)
            
            for update in client_updates:
                if 'model_update' in update:
                    weight = update['num_samples'] / total_samples
                    client_param = update['model_update'][name]
                    # Calculate difference from global model
                    diff = client_param - param.data
                    # Weight by local steps for FedNova
                    local_steps = update.get('local_steps', 1)
                    normalized_diff = diff * (local_steps / tau_eff)
                    aggregated_diff[name] += weight * normalized_diff
        
        # Apply momentum if enabled
        if self.algorithm_state['fednova_momentum'] is None:
            self.algorithm_state['fednova_momentum'] = {
                name: torch.zeros_like(param) 
                for name, param in self.global_model.named_parameters()
            }
        
        # Update global model with momentum
        for name, param in self.global_model.named_parameters():
            if name in aggregated_diff:
                # Apply momentum
                momentum = self.algorithm_state['fednova_momentum'][name]
                momentum.mul_(beta).add_(aggregated_diff[name], alpha=1.0)
                
                # Update parameters
                param.data.add_(momentum)
                
                # Store momentum for next round
                self.algorithm_state['fednova_momentum'][name] = momentum
    
    def evaluate_global_model(self, X_test, y_test) -> Dict:
        """Evaluate global model with comprehensive metrics"""
        eval_start_time = time.time()
        self.global_model.eval()
        
        # Prepare test data
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test).to(self.device),
            torch.LongTensor(y_test).to(self.device)
        )
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_dataloader:
                output = self.global_model(data)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = correct / total if total > 0 else 0.0
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Calculate AUC for binary classification
        try:
            if len(np.unique(all_targets)) == 2:
                auc = roc_auc_score(all_targets, [prob[1] for prob in all_probabilities])
            else:
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        eval_time = time.time() - eval_start_time
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'correct': correct,
            'total': total,
            'eval_time': eval_time,
            'privacy_preserved': self.privacy_enabled
        }
        
        # Store round metrics with enhanced information
        round_metric = {
            'round': self.algorithm_state['round'],
            'algorithm': self.aggregation_method,
            'round_time': self.round_times[-1] if self.round_times else 0.0,
            'aggregation_time': self.aggregation_times[-1] if self.aggregation_times else 0.0,
            'total_training_time': self.total_training_time,
            **metrics
        }
        self.round_metrics.append(round_metric)
        
        # Check for best performance and early stopping
        self._check_early_stopping(metrics)
        
        # Log metrics to experiment logger
        logger = get_experiment_logger()
        if logger:
            round_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'inference_time': eval_time
            }
            logger.log_round_metrics(self.algorithm_state['round'], round_metrics)
        
        return metrics
    
    def _check_early_stopping(self, current_metrics: Dict) -> None:
        """Check for early stopping based on F1-score improvement"""
        current_f1 = current_metrics['f1_score']
        
        # Update best performance if improved
        if current_f1 > self.best_f1_score + self.early_stopping_min_delta:
            self.best_f1_score = current_f1
            self.best_accuracy = current_metrics['accuracy']
            self.best_precision = current_metrics['precision']
            self.best_recall = current_metrics['recall']
            self.best_auc = current_metrics['auc']
            self.best_round = self.algorithm_state['round']
            self.best_metrics = current_metrics.copy()
            self.early_stopping_counter = 0
            
            print(f"   NEW BEST PERFORMANCE! F1: {self.best_f1_score:.4f} (Round {self.best_round})")
        else:
            self.early_stopping_counter += 1
            print(f"   No improvement for {self.early_stopping_counter} rounds")
            
        # Check if early stopping should be triggered
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.early_stopping_triggered = True
            print(f"   EARLY STOPPING TRIGGERED! No improvement for {self.early_stopping_patience} rounds")
    
    def should_stop_early(self) -> bool:
        """Check if early stopping condition is met"""
        return self.early_stopping_triggered
    
    def get_server_summary(self) -> Dict:
        """Get comprehensive server summary with all metrics"""
        return {
            'algorithm': self.aggregation_method,
            'total_rounds': self.algorithm_state['round'],
            'model': self.model_class.__name__,
            'device': self.device,
            'privacy_enabled': self.privacy_enabled,
            
            # Timing metrics
            'total_training_time': self.total_training_time,
            'total_aggregation_time': sum(self.aggregation_times),
            'avg_aggregation_time': np.mean(self.aggregation_times) if self.aggregation_times else 0.0,
            'total_round_time': sum(self.round_times),
            'avg_round_time': np.mean(self.round_times) if self.round_times else 0.0,
            
            # Early stopping metrics
            'early_stopping_triggered': self.early_stopping_triggered,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_counter': self.early_stopping_counter,
            
            # Best performance metrics
            'best_accuracy': self.best_accuracy,
            'best_f1_score': self.best_f1_score,
            'best_precision': self.best_precision,
            'best_recall': self.best_recall,
            'best_auc': self.best_auc,
            'best_round': self.best_round,
            'best_metrics': self.best_metrics,
            
            # Current performance (last round)
            'current_accuracy': self.round_metrics[-1]['accuracy'] if self.round_metrics else 0.0,
            'current_f1_score': self.round_metrics[-1]['f1_score'] if self.round_metrics else 0.0,
            'current_precision': self.round_metrics[-1]['precision'] if self.round_metrics else 0.0,
            'current_recall': self.round_metrics[-1]['recall'] if self.round_metrics else 0.0,
            'current_auc': self.round_metrics[-1]['auc'] if self.round_metrics else 0.0,
            
            # Complete metrics history
            'round_metrics_history': self.round_metrics,
            'client_metrics_history': self.client_metrics_history,
            
            # Statistics
            'total_clients_trained': len(set([client['client_id'] for client in self.client_metrics_history])),
            'avg_client_accuracy': np.mean([client['accuracy'] for client in self.client_metrics_history]) if self.client_metrics_history else 0.0,
            'avg_client_training_time': np.mean([client['training_time'] for client in self.client_metrics_history]) if self.client_metrics_history else 0.0,
            'avg_client_inference_time': np.mean([client['inference_time'] for client in self.client_metrics_history]) if self.client_metrics_history else 0.0
        }
