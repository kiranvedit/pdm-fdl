#!/usr/bin/env python3
"""
Phase 2B: Temporal Data Preparation for LSTM Integration
Creates time-series sequences from sensor data for temporal analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import json

def load_processed_data():
    """
    Load the processed multi-output data from Phase 1A
    """
    print("📥 Loading processed data from Phase 1A...")
    
    try:
        # Load arrays
        X_train = np.load('processed_data/X_train.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_binary_train = np.load('processed_data/y_binary_train.npy')
        y_binary_test = np.load('processed_data/y_binary_test.npy')
        y_type_train = np.load('processed_data/y_type_train.npy')
        y_type_test = np.load('processed_data/y_type_test.npy')
        
        # Load metadata
        with open('processed_data/feature_columns.txt', 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        with open('processed_data/failure_type_map.json', 'r') as f:
            failure_type_map = json.load(f)
            # Convert string keys to int
            failure_type_map = {int(k): v for k, v in failure_type_map.items()}
        
        import joblib
        scaler = joblib.load('processed_data/scaler.pkl')
        
        print("✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]:,}")
        print(f"   Test samples: {X_test.shape[0]:,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Feature columns: {feature_columns}")
        
        return X_train, X_test, y_binary_train, y_binary_test, y_type_train, y_type_test, feature_columns, failure_type_map, scaler
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        print("💡 Please run data_preparation_multi_output.py first")
        return None

def create_temporal_sequences(X, y_binary, y_type, sequence_length=10, overlap_ratio=0.5):
    """
    Create temporal sequences from tabular data for LSTM processing
    
    Args:
        X: Feature data (n_samples, n_features)
        y_binary: Binary targets (n_samples,)
        y_type: Multi-class targets (n_samples,)
        sequence_length: Length of each sequence
        overlap_ratio: Overlap between consecutive sequences (0.0 to 1.0)
    
    Returns:
        X_sequences: (n_sequences, sequence_length, n_features)
        y_binary_sequences: (n_sequences,)
        y_type_sequences: (n_sequences,)
    """
    print(f"\n🔄 Creating temporal sequences...")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Overlap ratio: {overlap_ratio}")
    
    n_samples, n_features = X.shape
    step_size = max(1, int(sequence_length * (1 - overlap_ratio)))
    
    # Calculate number of sequences
    n_sequences = (n_samples - sequence_length) // step_size + 1
    
    # Initialize arrays
    X_sequences = np.zeros((n_sequences, sequence_length, n_features))
    y_binary_sequences = np.zeros(n_sequences)
    y_type_sequences = np.zeros(n_sequences)
    
    # Create sequences
    for i in range(n_sequences):
        start_idx = i * step_size
        end_idx = start_idx + sequence_length
        
        if end_idx <= n_samples:
            X_sequences[i] = X[start_idx:end_idx]
            # Use the target at the end of the sequence (prediction target)
            y_binary_sequences[i] = y_binary[end_idx - 1]
            y_type_sequences[i] = y_type[end_idx - 1]
    
    print(f"✅ Created {n_sequences:,} sequences")
    print(f"   Original shape: {X.shape}")
    print(f"   Sequences shape: {X_sequences.shape}")
    print(f"   Step size: {step_size}")
    
    return X_sequences, y_binary_sequences.astype(int), y_type_sequences.astype(int)

def add_temporal_features(X_sequences):
    """
    Add temporal features to sequences (trends, rolling statistics, etc.)
    
    Args:
        X_sequences: (n_sequences, sequence_length, n_features)
    
    Returns:
        X_enhanced: (n_sequences, sequence_length, n_features + temporal_features)
    """
    print("\n🔧 Adding temporal features...")
    
    n_sequences, sequence_length, n_features = X_sequences.shape
    
    # Calculate temporal features
    temporal_features = []
    
    for seq_idx in range(n_sequences):
        seq_temporal = []
        
        for time_idx in range(sequence_length):
            time_features = []
            
            # Current values
            current_values = X_sequences[seq_idx, time_idx, :]
            
            # Rolling statistics (if we have enough history)
            if time_idx >= 2:
                # Rolling mean (last 3 time steps)
                window_start = max(0, time_idx - 2)
                window_data = X_sequences[seq_idx, window_start:time_idx+1, :]
                rolling_mean = np.mean(window_data, axis=0)
                rolling_std = np.std(window_data, axis=0)
                
                # Trend (current - previous)
                if time_idx > 0:
                    trend = current_values - X_sequences[seq_idx, time_idx-1, :]
                else:
                    trend = np.zeros_like(current_values)
                    
            else:
                rolling_mean = current_values
                rolling_std = np.zeros_like(current_values)
                trend = np.zeros_like(current_values)
            
            # Relative position in sequence
            position_feature = time_idx / (sequence_length - 1)
            
            # Combine temporal features
            time_features = np.concatenate([
                rolling_mean,     # Rolling mean of features
                rolling_std,      # Rolling std of features  
                trend,            # Trend (difference from previous)
                [position_feature] # Position in sequence
            ])
            
            seq_temporal.append(time_features)
        
        temporal_features.append(seq_temporal)
    
    temporal_features = np.array(temporal_features)
    
    # Combine original and temporal features
    X_enhanced = np.concatenate([X_sequences, temporal_features], axis=2)
    
    n_temporal_features = temporal_features.shape[2]
    print(f"✅ Added {n_temporal_features} temporal features per timestep")
    print(f"   Original features: {n_features}")
    print(f"   Total features: {X_enhanced.shape[2]}")
    print(f"   Enhanced shape: {X_enhanced.shape}")
    
    return X_enhanced

def simulate_sensor_noise_and_drift(X_sequences, noise_level=0.01, drift_probability=0.1):
    """
    Add realistic sensor noise and drift to make temporal patterns more realistic
    """
    print(f"\n🔧 Adding sensor noise and drift simulation...")
    print(f"   Noise level: {noise_level}")
    print(f"   Drift probability: {drift_probability}")
    
    X_noisy = X_sequences.copy()
    n_sequences, sequence_length, n_features = X_sequences.shape
    
    for seq_idx in range(n_sequences):
        # Add random noise
        noise = np.random.normal(0, noise_level, X_sequences[seq_idx].shape)
        X_noisy[seq_idx] += noise
        
        # Add sensor drift (gradual change over time)
        if np.random.random() < drift_probability:
            # Select random features to drift
            drift_features = np.random.choice(n_features, size=np.random.randint(1, 3), replace=False)
            
            for feat_idx in drift_features:
                # Create gradual drift
                drift_magnitude = np.random.uniform(-0.05, 0.05)
                drift_pattern = np.linspace(0, drift_magnitude, sequence_length)
                X_noisy[seq_idx, :, feat_idx] += drift_pattern
    
    print("✅ Sensor simulation applied")
    
    return X_noisy

def create_attention_masks(X_sequences, y_binary, attention_focus='failure_approach'):
    """
    Create attention masks to focus on important time periods
    
    Args:
        X_sequences: Temporal sequences
        y_binary: Binary failure labels
        attention_focus: 'failure_approach', 'uniform', or 'recent'
    """
    print(f"\n🎯 Creating attention masks ({attention_focus})...")
    
    n_sequences, sequence_length, _ = X_sequences.shape
    attention_masks = np.ones((n_sequences, sequence_length))
    
    if attention_focus == 'failure_approach':
        # Higher attention near failures
        for i in range(n_sequences):
            if y_binary[i] == 1:  # Failure case
                # Exponentially increasing attention towards end
                attention_weights = np.exp(np.linspace(0, 2, sequence_length))
                attention_masks[i] = attention_weights / np.sum(attention_weights) * sequence_length
    
    elif attention_focus == 'recent':
        # Higher attention on recent timesteps
        for i in range(n_sequences):
            attention_weights = np.exp(np.linspace(-1, 1, sequence_length))
            attention_masks[i] = attention_weights / np.sum(attention_weights) * sequence_length
    
    # attention_focus == 'uniform' uses default uniform weights
    
    print(f"✅ Attention masks created")
    print(f"   Shape: {attention_masks.shape}")
    print(f"   Mean attention weight: {np.mean(attention_masks):.3f}")
    
    return attention_masks

def analyze_temporal_patterns(X_sequences, y_binary, y_type, feature_names):
    """
    Analyze temporal patterns in the sequences
    """
    print("\n📊 Analyzing temporal patterns...")
    
    n_sequences, sequence_length, n_features = X_sequences.shape
    
    # Create visualizations directory
    os.makedirs('visualizations/temporal', exist_ok=True)
    
    # Plot 1: Average feature trends for failure vs no-failure
    plt.figure(figsize=(15, 10))
    
    # Separate failure and no-failure sequences
    failure_mask = y_binary == 1
    no_failure_mask = y_binary == 0
    
    failure_sequences = X_sequences[failure_mask]
    no_failure_sequences = X_sequences[no_failure_mask]
    
    print(f"   Failure sequences: {len(failure_sequences)}")
    print(f"   No-failure sequences: {len(no_failure_sequences)}")
    
    # Plot feature trends for original features only
    original_features = min(6, n_features)  # First 6 are original features
    
    for i in range(original_features):
        plt.subplot(2, 3, i + 1)
        
        if len(failure_sequences) > 0:
            failure_mean = np.mean(failure_sequences[:, :, i], axis=0)
            failure_std = np.std(failure_sequences[:, :, i], axis=0)
            time_steps = range(sequence_length)
            
            plt.plot(time_steps, failure_mean, 'r-', label='Failure', linewidth=2)
            plt.fill_between(time_steps, 
                           failure_mean - failure_std, 
                           failure_mean + failure_std, 
                           alpha=0.3, color='red')
        
        if len(no_failure_sequences) > 0:
            no_failure_mean = np.mean(no_failure_sequences[:, :, i], axis=0)
            no_failure_std = np.std(no_failure_sequences[:, :, i], axis=0)
            
            plt.plot(time_steps, no_failure_mean, 'b-', label='No Failure', linewidth=2)
            plt.fill_between(time_steps, 
                           no_failure_mean - no_failure_std, 
                           no_failure_mean + no_failure_std, 
                           alpha=0.3, color='blue')
        
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        plt.title(f'{feature_name}')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/temporal/temporal_feature_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Sequence length distribution analysis
    plt.figure(figsize=(12, 8))
    
    # Plot distributions by failure type
    failure_types = np.unique(y_type)
    
    plt.subplot(2, 2, 1)
    for failure_type in failure_types:
        if failure_type == 0:
            continue  # Skip no-failure for this analysis
        
        type_mask = y_type == failure_type
        if np.sum(type_mask) > 0:
            type_sequences = X_sequences[type_mask]
            # Calculate sequence variance as a complexity measure
            sequence_variance = np.var(type_sequences, axis=1).mean(axis=1)
            plt.hist(sequence_variance, alpha=0.6, label=f'Type {failure_type}', bins=20)
    
    plt.title('Sequence Complexity by Failure Type')
    plt.xlabel('Average Variance per Sequence')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot 3: Feature correlation over time
    plt.subplot(2, 2, 2)
    
    # Calculate correlation at each timestep
    timestep_correlations = []
    for t in range(sequence_length):
        timestep_data = X_sequences[:, t, :original_features]
        if timestep_data.shape[1] > 1:
            corr_matrix = np.corrcoef(timestep_data.T)
            # Average correlation (excluding diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = np.mean(np.abs(corr_matrix[mask]))
            timestep_correlations.append(avg_corr)
    
    plt.plot(range(sequence_length), timestep_correlations, 'g-', linewidth=2)
    plt.title('Feature Correlation Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Average Absolute Correlation')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Temporal attention importance
    plt.subplot(2, 2, 3)
    
    # Calculate feature importance over time (variance-based)
    feature_importance_over_time = np.var(X_sequences, axis=0)  # (sequence_length, n_features)
    
    for i in range(min(3, original_features)):  # Plot top 3 features
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        plt.plot(range(sequence_length), feature_importance_over_time[:, i], 
                label=feature_name, linewidth=2)
    
    plt.title('Feature Variance Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Failure prediction timeline
    plt.subplot(2, 2, 4)
    
    if len(failure_sequences) > 0:
        # Show how features evolve leading up to failure
        sample_failure_seq = failure_sequences[0]  # Take first failure sequence
        
        for i in range(min(3, original_features)):
            feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
            plt.plot(range(sequence_length), sample_failure_seq[:, i], 
                    'o-', label=feature_name, linewidth=2, markersize=4)
        
        plt.axvline(x=sequence_length-1, color='red', linestyle='--', 
                   label='Failure Point', alpha=0.7)
        plt.title('Sample Failure Evolution')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/temporal/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Temporal analysis visualizations saved to 'visualizations/temporal/'")

def save_temporal_data(X_train_seq, X_test_seq, y_binary_train_seq, y_binary_test_seq, 
                      y_type_train_seq, y_type_test_seq, attention_masks_train, attention_masks_test,
                      feature_names, sequence_length):
    """
    Save processed temporal data
    """
    print("\n💾 Saving temporal data...")
    
    # Create directory
    os.makedirs('processed_data/temporal', exist_ok=True)
    
    # Save temporal sequences
    np.save('processed_data/temporal/X_train_sequences.npy', X_train_seq)
    np.save('processed_data/temporal/X_test_sequences.npy', X_test_seq)
    np.save('processed_data/temporal/y_binary_train_sequences.npy', y_binary_train_seq)
    np.save('processed_data/temporal/y_binary_test_sequences.npy', y_binary_test_seq)
    np.save('processed_data/temporal/y_type_train_sequences.npy', y_type_train_seq)
    np.save('processed_data/temporal/y_type_test_sequences.npy', y_type_test_seq)
    np.save('processed_data/temporal/attention_masks_train.npy', attention_masks_train)
    np.save('processed_data/temporal/attention_masks_test.npy', attention_masks_test)
    
    # Save metadata
    temporal_metadata = {
        'sequence_length': sequence_length,
        'n_features': X_train_seq.shape[2],
        'n_train_sequences': X_train_seq.shape[0],
        'n_test_sequences': X_test_seq.shape[0],
        'enhanced_feature_names': feature_names
    }
    
    with open('processed_data/temporal/temporal_metadata.json', 'w') as f:
        json.dump(temporal_metadata, f, indent=2)
    
    print("✅ Temporal data saved successfully!")
    print(f"   Training sequences: {X_train_seq.shape}")
    print(f"   Test sequences: {X_test_seq.shape}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Features per timestep: {X_train_seq.shape[2]}")

def main():
    """
    Main function for Phase 2B temporal data preparation
    """
    print("=" * 80)
    print("🚀 PHASE 2B: TEMPORAL DATA PREPARATION FOR LSTM INTEGRATION")
    print("   Creating Time-Series Sequences for Temporal Analysis")
    print("   Hybrid CNN-LSTM Architecture Data Pipeline")
    print("=" * 80)
    
    # Step 1: Load processed data from Phase 1A
    data = load_processed_data()
    if data is None:
        return
    
    X_train, X_test, y_binary_train, y_binary_test, y_type_train, y_type_test, feature_columns, failure_type_map, scaler = data
    
    # Step 2: Create temporal sequences
    sequence_length = 10  # 10 timesteps for temporal analysis
    overlap_ratio = 0.5   # 50% overlap between sequences
    
    print(f"\n🔄 Processing training data...")
    X_train_seq, y_binary_train_seq, y_type_train_seq = create_temporal_sequences(
        X_train, y_binary_train, y_type_train, sequence_length, overlap_ratio
    )
    
    print(f"\n🔄 Processing test data...")
    X_test_seq, y_binary_test_seq, y_type_test_seq = create_temporal_sequences(
        X_test, y_binary_test, y_type_test, sequence_length, overlap_ratio
    )
    
    # Step 3: Add temporal features
    print(f"\n🔧 Enhancing with temporal features...")
    X_train_enhanced = add_temporal_features(X_train_seq)
    X_test_enhanced = add_temporal_features(X_test_seq)
    
    # Step 4: Add sensor noise simulation
    X_train_realistic = simulate_sensor_noise_and_drift(X_train_enhanced, noise_level=0.01)
    X_test_realistic = simulate_sensor_noise_and_drift(X_test_enhanced, noise_level=0.005)  # Less noise in test
    
    # Step 5: Create attention masks
    attention_masks_train = create_attention_masks(X_train_realistic, y_binary_train_seq, 'failure_approach')
    attention_masks_test = create_attention_masks(X_test_realistic, y_binary_test_seq, 'failure_approach')
    
    # Step 6: Analyze temporal patterns
    enhanced_feature_names = feature_columns + [f'{col}_rolling_mean' for col in feature_columns] + \
                           [f'{col}_rolling_std' for col in feature_columns] + \
                           [f'{col}_trend' for col in feature_columns] + ['position_in_sequence']
    
    analyze_temporal_patterns(X_train_realistic, y_binary_train_seq, y_type_train_seq, enhanced_feature_names)
    
    # Step 7: Save temporal data
    save_temporal_data(X_train_realistic, X_test_realistic, 
                      y_binary_train_seq, y_binary_test_seq,
                      y_type_train_seq, y_type_test_seq,
                      attention_masks_train, attention_masks_test,
                      enhanced_feature_names, sequence_length)
    
    print("\n" + "=" * 80)
    print("✅ PHASE 2B: TEMPORAL DATA PREPARATION COMPLETED!")
    print(f"🎯 Temporal Dataset Ready:")
    print(f"   📊 Training sequences: {X_train_realistic.shape[0]:,}")
    print(f"   📊 Test sequences: {X_test_realistic.shape[0]:,}")
    print(f"   ⏱️ Sequence length: {sequence_length}")
    print(f"   📊 Features per timestep: {X_train_realistic.shape[2]}")
    print(f"   🎯 Attention masks: ✅")
    print(f"   🔧 Temporal features: ✅")
    print(f"   📈 Sensor simulation: ✅")
    print("   Ready for CNN-LSTM hybrid model training!")
    print("=" * 80)

if __name__ == "__main__":
    main()
