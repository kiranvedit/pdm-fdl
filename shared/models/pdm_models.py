#!/usr/bin/env python3
"""
Shared Model Definitions for Industrial Predictive Maintenance
Provides CNN, LSTM, and Hybrid model architectures for both central and federated learning
"""

import torch
import torch.nn as nn
import numpy as np

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class CentralCNNModel(nn.Module):
    """CNN Model for Industrial Predictive Maintenance"""
    
    def __init__(self, input_dim=5, num_classes=6, 
                 conv_filters=[32, 64, 128], 
                 fc_hidden=[128, 64], 
                 dropout_rate=0.3):
        super(CentralCNNModel, self).__init__()
        
        # CNN layers
        layers = []
        in_channels = 1
        
        for i, filters in enumerate(conv_filters):
            layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            if i < len(conv_filters) - 1:
                layers.append(nn.MaxPool1d(2))
            in_channels = filters
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Fully connected layers
        fc_layers = []
        in_features = conv_filters[-1]
        
        for hidden_dim in fc_hidden:
            fc_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_dim
        
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CentralLSTMModel(nn.Module):
    """LSTM Model for Industrial Predictive Maintenance"""
    
    def __init__(self, input_dim=5, num_classes=6,
                 hidden_dim=64, num_layers=2, 
                 dropout_rate=0.3, bidirectional=True):
        super(CentralLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # Reshape for LSTM if needed
        if len(x.shape) == 2:
            # Create sequences from features
            batch_size = x.shape[0]
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(attended_output)
        return output

class CentralHybridModel(nn.Module):
    """Hybrid CNN-LSTM Model for Industrial Predictive Maintenance"""
    
    def __init__(self, input_dim=5, num_classes=6,
                 cnn_filters=[32, 64], lstm_hidden=64,
                 dropout_rate=0.3):
        super(CentralHybridModel, self).__init__()
        
        # CNN branch for spatial features
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, cnn_filters[0], 3, padding=1),
            nn.BatchNorm1d(cnn_filters[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(cnn_filters[0], cnn_filters[1], 3, padding=1),
            nn.BatchNorm1d(cnn_filters[1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # LSTM branch for temporal features
        self.lstm_branch = nn.LSTM(
            input_dim, lstm_hidden, 
            batch_first=True, dropout=dropout_rate
        )
        
        # Fusion and classification
        fusion_input_dim = cnn_filters[1] + lstm_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN features
        cnn_input = x.unsqueeze(1) if len(x.shape) == 2 else x
        cnn_features = self.cnn_branch(cnn_input)
        
        # LSTM features
        lstm_input = x.unsqueeze(1) if len(x.shape) == 2 else x
        lstm_out, _ = self.lstm_branch(lstm_input)
        lstm_features = lstm_out[:, -1, :]  # Last timestep
        
        # Fusion
        combined = torch.cat([cnn_features, lstm_features], dim=1)
        output = self.fusion(combined)
        return output
