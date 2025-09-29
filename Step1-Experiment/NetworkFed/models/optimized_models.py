"""
Optimized Model Definitions with Best Hyperparameters
=====================================================
This module contains the optimized model definitions with the best hyperparameters
found during hyperparameter tuning in Step 1A.

Models included:
- OptimizedCNNModel: Best accuracy 93.95%
- OptimizedLSTMModel: Best accuracy 94.15% 
- OptimizedHybridModel: Best accuracy 92.20%

Generated on: August 1, 2025
Ready for: Federated Learning Implementation (Phase 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

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

def create_optimized_model(model_type: str, input_dim: int = 10, num_classes: int = 2, sequence_length: int = 10):
    """
    Factory function to create optimized models with best hyperparameters

    Args:
        model_type: 'cnn', 'lstm', or 'hybrid'
        input_dim: Number of input features
        num_classes: Number of output classes
        sequence_length: Sequence length for LSTM models

    Returns:
        Optimized model instance
    """
    model_map = {
        'cnn': OptimizedCNNModel,
        'lstm': OptimizedLSTMModel,
        'hybrid': OptimizedHybridModel
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(model_map.keys())}")

    if model_type == 'cnn':
        return model_map[model_type](input_dim=input_dim, num_classes=num_classes)
    else:
        return model_map[model_type](input_dim=input_dim, num_classes=num_classes, sequence_length=sequence_length)

# Training configuration for optimized models
OPTIMIZED_TRAINING_CONFIG = {
    'cnn': {
        'batch_size': 32,
        'learning_rate': 0.0005,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'step_size': 10,
        'gamma': 0.5,
        'weight_decay': 1e-4
    },
    'lstm': {
        'batch_size': 16,
        'learning_rate': 0.0005,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'step_size': 10,
        'gamma': 0.5,
        'weight_decay': 1e-4
    },
    'hybrid': {
        'batch_size': 16,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'step_size': 10,
        'gamma': 0.5,
        'weight_decay': 1e-4
    }
}

# Best hyperparameters summary
BEST_HYPERPARAMETERS = {
    'cnn': {
        'batch_size': 32,
        'conv_filters': [32, 64, 128],
        'dropout_rate': 0.3,
        'fc_hidden': [256, 128],
        'learning_rate': 0.0005,
        # 'test_accuracy': REMOVED - no hardcoded values allowed
    },
    'lstm': {
        'batch_size': 16,
        'bidirectional': True,
        'dropout_rate': 0.2,
        'hidden_dim': 64,
        'learning_rate': 0.0005,
        'num_layers': 1,
        # 'test_accuracy': REMOVED - no hardcoded values allowed
    },
    'hybrid': {
        'batch_size': 16,
        'cnn_filters': [32, 64],
        'dropout_rate': 0.4,
        'learning_rate': 0.001,
        'lstm_hidden': 128,
        # 'test_accuracy': REMOVED - no hardcoded values allowed
    }
}
