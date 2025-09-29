"""
Federated Learning Client Implementation
Supports all algorithms with comprehensive metrics tracking and privacy preservation
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional


class FederatedClient:
    """
    Federated learning client supporting all algorithms with comprehensive metrics tracking
    Handles algorithm-specific training procedures and privacy preservation
    
    Enhanced Features:
    - Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUC
    - Training time tracking: Per-epoch, total training time
    - Inference time tracking: Total and per-sample inference times
    - Client-level performance analysis
    """
    
    def __init__(self, client_id: str, local_data: Tuple, model_class, model_params: Dict,
                 local_epochs: int = 5, batch_size: int = 16, learning_rate: float = 0.001,
                 device: str = 'cpu'):
        self.client_id = client_id
        self.X_local, self.y_local = local_data
        self.model_class = model_class
        self.model_params = model_params
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize local model
        self.model = model_class(**model_params).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Enhanced metrics tracking
        self.training_history = []
        self.current_metrics = {}
        self.inference_times = []
        
        print(f"Client {client_id} initialized with {len(self.X_local)} samples")
    
    def local_training(self, global_model_state: Dict, algorithm: str = 'fedavg',
                      algorithm_kwargs: Optional[Dict] = None, variable_epochs: bool = False) -> Tuple[Dict, Dict]:
        """
        Perform local training with algorithm-specific configurations and comprehensive metrics
        
        Returns:
            model_update: Updated model parameters
            training_metrics: Enhanced training statistics and metadata
        """
        algorithm_kwargs = algorithm_kwargs or {}
        
        # Load global model (use strict=False to handle BatchNorm parameters)
        self.model.load_state_dict(global_model_state, strict=False)
        
        # Algorithm-specific configurations
        if variable_epochs and algorithm == 'fednova':
            # Simulate system heterogeneity with variable local epochs
            actual_epochs = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        else:
            actual_epochs = self.local_epochs
        
        # Store initial weights for FedProx
        initial_weights = None
        mu = 0.01
        if algorithm == 'fedprox':
            initial_weights = {name: param.clone().detach() 
                             for name, param in self.model.named_parameters()}
            mu = algorithm_kwargs.get('mu', 0.01)
        
        # Prepare data
        dataset = TensorDataset(
            torch.FloatTensor(self.X_local).to(self.device),
            torch.LongTensor(self.y_local).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop with enhanced metrics
        self.model.train()
        total_loss = 0.0
        steps_taken = 0
        epoch_times = []
        start_time = time.time()
        
        for epoch in range(actual_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Standard loss
                loss = F.cross_entropy(output, target)
                
                # Algorithm-specific loss modifications
                if algorithm == 'fedprox' and initial_weights is not None:
                    # Add proximal regularization term
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in initial_weights:
                            current_param = param.view(-1)
                            initial_param = initial_weights[name].view(-1)
                            proximal_term += torch.sum((current_param - initial_param) ** 2)
                    
                    loss += (mu / 2.0) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                steps_taken += 1
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
        
        training_time = time.time() - start_time
        
        # Calculate comprehensive local metrics
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_start = time.time()
        
        with torch.no_grad():
            for data, target in dataloader:
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        inference_time = time.time() - inference_start
        sample_inference_time = inference_time / len(self.X_local) if len(self.X_local) > 0 else 0.0
        self.inference_times.append(inference_time)
        
        # Calculate enhanced metrics
        accuracy = accuracy_score(all_targets, all_predictions)
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
        
        # Get model update (only learnable parameters)
        model_update = {name: param.data.clone() 
                       for name, param in self.model.named_parameters()}
        
        # Store enhanced metrics
        self.current_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': total_loss / max(1, steps_taken),
            'epochs': actual_epochs,
            'steps': steps_taken,
            'training_time': training_time,
            'inference_time': inference_time,
            'sample_inference_time': sample_inference_time,
            'avg_epoch_time': np.mean(epoch_times)
        }
        
        # Enhanced training metrics for server
        training_metrics = {
            'client_id': self.client_id,
            'num_samples': len(self.X_local),
            'local_accuracy': accuracy,
            'local_precision': precision,
            'local_recall': recall,
            'local_f1_score': f1,
            'local_auc': auc,
            'local_loss': total_loss / max(1, steps_taken),
            'local_steps': steps_taken,
            'local_epochs': actual_epochs,
            'training_time': training_time,
            'inference_time': inference_time,
            'sample_inference_time': sample_inference_time,
            'algorithm': algorithm,
            'privacy_preserved': True,
            'model_update': model_update
        }
        
        # Store enhanced training history
        self.training_history.append({
            'round': len(self.training_history) + 1,
            'algorithm': algorithm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': total_loss / max(1, steps_taken),
            'epochs': actual_epochs,
            'steps': steps_taken,
            'training_time': training_time,
            'inference_time': inference_time,
            'sample_inference_time': sample_inference_time
        })
        
        print(f"Client {self.client_id} training complete:")
        print(f"   Algorithm: {algorithm.upper()}")
        print(f"   Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"   Loss: {total_loss / max(1, steps_taken):.4f}, Epochs: {actual_epochs}")
        print(f"   Training time: {training_time:.2f}s, Inference time: {inference_time:.4f}s")
        print(f"   Sample inference time: {sample_inference_time:.6f}s")
        
        if algorithm == 'fedprox' and initial_weights is not None:
            print(f"   FedProx mu: {mu}")
        
        return model_update, training_metrics
    
    def get_client_summary(self) -> Dict:
        """Get enhanced client summary"""
        return {
            'client_id': self.client_id,
            'num_samples': len(self.X_local),
            'accuracy': self.current_metrics.get('accuracy', 0.0),
            'precision': self.current_metrics.get('precision', 0.0),
            'recall': self.current_metrics.get('recall', 0.0),
            'f1_score': self.current_metrics.get('f1_score', 0.0),
            'auc': self.current_metrics.get('auc', 0.0),
            'loss': self.current_metrics.get('loss', 0.0),
            'epochs': self.current_metrics.get('epochs', 0),
            'steps': self.current_metrics.get('steps', 0),
            'training_time': self.current_metrics.get('training_time', 0.0),
            'inference_time': self.current_metrics.get('inference_time', 0.0),
            'sample_inference_time': self.current_metrics.get('sample_inference_time', 0.0),
            'privacy_preserved': True,
            'total_inference_time': sum(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0
        }
    
    def local_inference(self, test_dataloader) -> Dict:
        """Perform local inference with enhanced metrics"""
        inference_start = time.time()
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                
                loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        inference_time = time.time() - inference_start
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        try:
            if len(np.unique(all_targets)) == 2:
                auc = roc_auc_score(all_targets, [prob[1] for prob in all_probabilities])
            else:
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        avg_loss = total_loss / len(test_dataloader) if len(test_dataloader) > 0 else 0.0
        sample_inference_time = inference_time / len(all_targets) if len(all_targets) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': avg_loss,
            'samples': len(all_targets),
            'inference_time': inference_time,
            'sample_inference_time': sample_inference_time
        }
