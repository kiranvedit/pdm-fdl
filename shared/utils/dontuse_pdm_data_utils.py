#!/usr/bin/env python3
"""
Shared Data Utilities for Industrial Predictive Maintenance
Provides common data loading and preprocessing functions for both central and federated learning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Set random seed
SEED = 42
np.random.seed(SEED)

def load_and_prepare_dataset(file_path="data/ai4i2020.csv"):
    """Load and prepare the industrial dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("⚠️ Dataset not found, creating synthetic data...")
        df = create_synthetic_data()
    
    # Data preprocessing
    # Handle missing values
    df.ffill(inplace=True)
    
    # Map failure types
    def map_failure_type(row):
        if row['TWF'] == 1: return 1
        elif row['HDF'] == 1: return 2
        elif row['PWF'] == 1: return 3
        elif row['OSF'] == 1: return 4
        elif row['RNF'] == 1: return 5
        return 0
    
    df['Failure_Type'] = df.apply(map_failure_type, axis=1)
    
    # Select features
    features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    
    X = df[features].values
    y = df['Failure_Type'].values
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y)}")
    
    return X_scaled, y, scaler

def create_synthetic_data(n_samples=10000):
    """Create synthetic industrial data"""
    np.random.seed(SEED)
    
    data = {
        'Air temperature [K]': np.random.normal(300, 10, n_samples),
        'Process temperature [K]': np.random.normal(310, 15, n_samples),
        'Rotational speed [rpm]': np.random.normal(1500, 300, n_samples),
        'Torque [Nm]': np.random.normal(40, 10, n_samples),
        'Tool wear [min]': np.random.exponential(50, n_samples),
        'TWF': np.random.binomial(1, 0.1, n_samples),
        'HDF': np.random.binomial(1, 0.05, n_samples),
        'PWF': np.random.binomial(1, 0.03, n_samples),
        'OSF': np.random.binomial(1, 0.02, n_samples),
        'RNF': np.random.binomial(1, 0.01, n_samples)
    }
    
    return pd.DataFrame(data)
