#!/usr/bin/env python3
"""
Phase 1A Enhanced: Multi-Output Data Preparation Script
Download and preprocess the AI4I 2020 Predictive Maintenance Dataset
for both Machine Failure Detection AND Failure Type Classification
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import kagglehub

def download_dataset():
    """
    Download the AI4I 2020 dataset using kagglehub
    """
    print("🔽 Downloading AI4I 2020 Predictive Maintenance Dataset...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def load_and_explore_data(dataset_path):
    """
    Load and explore the dataset
    """
    print("\n📊 Loading and exploring dataset...")
    
    # Find CSV files in the dataset directory
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("❌ No CSV files found in dataset")
        return None
    
    # Load the dataset (assuming the first CSV file is our target)
    data_file = csv_files[0]
    print(f"📋 Loading data from: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Display basic information
    print(f"\n📈 Dataset Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"\n🔍 First 5 rows:")
    print(df.head())
    
    print(f"\n📊 Data Info:")
    print(df.info())
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\n❓ Missing Values:")
    print(missing_values[missing_values > 0])
    
    # Analyze target distributions
    if 'Machine failure' in df.columns:
        target_dist = df['Machine failure'].value_counts()
        print(f"\n🎯 Target Distribution (Machine failure):")
        print(target_dist)
        print(f"📊 Class Balance Ratio: {target_dist.iloc[1] / target_dist.iloc[0]:.3f}")
    
    # Analyze failure types
    failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    existing_failure_cols = [col for col in failure_columns if col in df.columns]
    
    if existing_failure_cols:
        print(f"\n🔍 Failure Types Analysis:")
        for col in existing_failure_cols:
            count = df[col].sum()
            print(f"   {col}: {count} cases ({count/len(df)*100:.2f}%)")
        
        # Check overlap between failure types
        failure_mask = df[existing_failure_cols].sum(axis=1) > 0
        machine_failure_true = df['Machine failure'] == 1
        
        print(f"\n📊 Failure Analysis:")
        print(f"   Machine failures: {machine_failure_true.sum()}")
        print(f"   Specific failure types: {failure_mask.sum()}")
        print(f"   Overlap ratio: {(failure_mask & machine_failure_true).sum() / machine_failure_true.sum():.3f}")
        
        # Create failure type labels
        df['failure_type'] = 'No_Failure'
        for i, col in enumerate(existing_failure_cols):
            df.loc[df[col] == 1, 'failure_type'] = col
        
        print(f"\n🏷️ Failure Type Distribution:")
        print(df['failure_type'].value_counts())
    
    return df

def preprocess_data_multi_output(df):
    """
    Preprocess the dataset for multi-output machine learning
    Output 1: Binary classification (Machine failure: 0/1)
    Output 2: Multi-class classification (Failure type: No_Failure, TWF, HDF, PWF, OSF, RNF)
    """
    print("\n🔧 Preprocessing data for multi-output prediction...")
    
    # Create a copy for processing
    processed_df = df.copy()
    
    # Remove unnecessary columns if they exist
    columns_to_remove = ['UDI', 'Product ID'] if 'UDI' in processed_df.columns else []
    if columns_to_remove:
        processed_df = processed_df.drop(columns=columns_to_remove)
        print(f"🗑️ Removed columns: {columns_to_remove}")
    
    # Handle categorical variables
    if 'Type' in processed_df.columns:
        le = LabelEncoder()
        processed_df['Type_encoded'] = le.fit_transform(processed_df['Type'])
        processed_df = processed_df.drop('Type', axis=1)
        print("🔤 Encoded 'Type' column")
    
    # Prepare features (X)
    feature_columns = ['Air temperature [K]', 'Process temperature [K]', 
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_encoded']
    
    X = processed_df[feature_columns].copy()
    
    # Prepare target 1: Binary machine failure
    y_binary = processed_df['Machine failure'].copy()
    
    # Prepare target 2: Failure type classification
    failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # Create failure type labels
    y_failure_type = np.zeros(len(processed_df), dtype=int)  # 0 = No_Failure
    failure_type_map = {0: 'No_Failure', 1: 'TWF', 2: 'HDF', 3: 'PWF', 4: 'OSF', 5: 'RNF'}
    
    for i, col in enumerate(failure_columns):
        if col in processed_df.columns:
            mask = processed_df[col] == 1
            y_failure_type[mask] = i + 1  # 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Binary target shape: {y_binary.shape}")
    print(f"✅ Failure type target shape: {y_failure_type.shape}")
    print(f"📋 Feature columns: {list(X.columns)}")
    
    # Print failure type distribution
    unique_types, counts = np.unique(y_failure_type, return_counts=True)
    print(f"\n📊 Failure Type Distribution:")
    for type_id, count in zip(unique_types, counts):
        print(f"   {failure_type_map[type_id]}: {count} samples")
    
    return X, y_binary, y_failure_type, failure_type_map

def apply_advanced_balancing(X, y_binary, y_failure_type):
    """
    Apply advanced balancing for multi-output prediction
    """
    print("\n⚖️ Applying advanced balancing for multi-output prediction...")
    
    # Check original distributions
    print(f"📊 Original Binary Distribution:")
    unique, counts = np.unique(y_binary, return_counts=True)
    print(f"   {dict(zip(unique, counts))}")
    
    print(f"📊 Original Failure Type Distribution:")
    unique, counts = np.unique(y_failure_type, return_counts=True)
    print(f"   {dict(zip(unique, counts))}")
    
    # Strategy 1: Balance binary classification first
    # Only apply SMOTE to machine failure cases
    failure_mask = y_binary == 1
    no_failure_mask = y_binary == 0
    
    # Get failure cases
    X_failures = X[failure_mask]
    y_binary_failures = y_binary[failure_mask]
    y_type_failures = y_failure_type[failure_mask]
    
    # Get a balanced sample of no-failure cases
    X_no_failures = X[no_failure_mask]
    y_binary_no_failures = y_binary[no_failure_mask]
    y_type_no_failures = y_failure_type[no_failure_mask]
    
    # Sample same number of no-failure cases as we'll have failure cases after SMOTE
    n_failure_samples = len(X_failures)
    
    # Apply SMOTE to failure cases to balance failure types
    if len(np.unique(y_type_failures)) > 1:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, len(X_failures)-1))
            X_failures_balanced, y_type_failures_balanced = smote.fit_resample(X_failures, y_type_failures)
            y_binary_failures_balanced = np.ones(len(X_failures_balanced))
            
            print(f"✅ SMOTE applied to failure cases:")
            print(f"   Original failures: {len(X_failures)}")
            print(f"   Balanced failures: {len(X_failures_balanced)}")
            
        except Exception as e:
            print(f"⚠️ SMOTE failed, using original failure data: {e}")
            X_failures_balanced = X_failures
            y_binary_failures_balanced = y_binary_failures
            y_type_failures_balanced = y_type_failures
    else:
        X_failures_balanced = X_failures
        y_binary_failures_balanced = y_binary_failures
        y_type_failures_balanced = y_type_failures
    
    # Balance with no-failure cases
    n_balanced_failures = len(X_failures_balanced)
    n_no_failures_to_sample = min(n_balanced_failures, len(X_no_failures))
    
    # Randomly sample no-failure cases
    np.random.seed(42)
    no_failure_indices = np.random.choice(len(X_no_failures), n_no_failures_to_sample, replace=False)
    
    X_no_failures_sampled = X_no_failures.iloc[no_failure_indices]
    y_binary_no_failures_sampled = y_binary_no_failures.iloc[no_failure_indices]
    y_type_no_failures_sampled = y_type_no_failures[no_failure_indices]
    
    # Combine balanced datasets
    X_balanced = pd.concat([X_failures_balanced, X_no_failures_sampled], ignore_index=True)
    y_binary_balanced = np.concatenate([y_binary_failures_balanced, y_binary_no_failures_sampled])
    y_type_balanced = np.concatenate([y_type_failures_balanced, y_type_no_failures_sampled])
    
    # Shuffle the combined dataset
    shuffle_indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced.iloc[shuffle_indices].reset_index(drop=True)
    y_binary_balanced = y_binary_balanced[shuffle_indices]
    y_type_balanced = y_type_balanced[shuffle_indices]
    
    # Check new distributions
    print(f"\n📊 Balanced Binary Distribution:")
    unique, counts = np.unique(y_binary_balanced, return_counts=True)
    print(f"   {dict(zip(unique, counts))}")
    
    print(f"📊 Balanced Failure Type Distribution:")
    unique, counts = np.unique(y_type_balanced, return_counts=True)
    print(f"   {dict(zip(unique, counts))}")
    
    print(f"✅ Final balanced dataset shape: {X_balanced.shape}")
    
    return X_balanced, y_binary_balanced, y_type_balanced

def create_train_test_split_multi_output(X, y_binary, y_failure_type, test_size=0.2, random_state=42):
    """
    Create train-test split for multi-output data
    """
    print(f"\n✂️ Creating train-test split for multi-output ({int((1-test_size)*100)}% / {int(test_size*100)}%)...")
    
    # Use binary target for stratification (more balanced)
    X_train, X_test, y_binary_train, y_binary_test, y_type_train, y_type_test = train_test_split(
        X, y_binary, y_failure_type, test_size=test_size, random_state=random_state, stratify=y_binary
    )
    
    print(f"📈 Training set: {X_train.shape}")
    print(f"📈 Test set: {X_test.shape}")
    
    # Check distributions in train/test
    print(f"\n📊 Training Set Distributions:")
    print(f"   Binary: {dict(zip(*np.unique(y_binary_train, return_counts=True)))}")
    print(f"   Failure Type: {dict(zip(*np.unique(y_type_train, return_counts=True)))}")
    
    print(f"\n📊 Test Set Distributions:")
    print(f"   Binary: {dict(zip(*np.unique(y_binary_test, return_counts=True)))}")
    print(f"   Failure Type: {dict(zip(*np.unique(y_type_test, return_counts=True)))}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features normalized using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_binary_train, y_binary_test, y_type_train, y_type_test, scaler

def visualize_multi_output_data(df, X, y_binary, y_failure_type, failure_type_map):
    """
    Create visualizations for the multi-output dataset
    """
    print("\n📊 Creating multi-output data visualizations...")
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Create comprehensive visualization
    plt.figure(figsize=(16, 12))
    
    # 1. Binary target distribution
    plt.subplot(3, 4, 1)
    y_binary.value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Binary Target Distribution\n(Machine Failure)')
    plt.xlabel('Machine Failure')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # 2. Failure type distribution
    plt.subplot(3, 4, 2)
    failure_type_labels = [failure_type_map[i] for i in y_failure_type]
    pd.Series(failure_type_labels).value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Failure Type Distribution')
    plt.xlabel('Failure Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 3. Feature correlations
    plt.subplot(3, 4, 3)
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, 
                cbar_kws={'shrink': 0.8}, fmt='.2f')
    plt.title('Feature Correlations')
    
    # 4. Feature distributions
    feature_cols = ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    colors = ['green', 'orange', 'purple', 'red', 'brown']
    
    for i, (col, color) in enumerate(zip(feature_cols, colors)):
        if col in df.columns:
            plt.subplot(3, 4, 4 + i)
            df[col].hist(bins=30, alpha=0.7, color=color)
            plt.title(f'{col.split()[0]} Distribution')
            plt.xlabel(col)
            plt.ylabel('Frequency')
    
    # 9. Failure types vs Machine failures
    plt.subplot(3, 4, 9)
    failure_cross = pd.crosstab(y_binary, failure_type_labels)
    failure_cross.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Machine Failure vs Failure Types')
    plt.xlabel('Machine Failure')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 10-12. Feature distributions by failure type
    for i, col in enumerate(['Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]']):
        if col in df.columns:
            plt.subplot(3, 4, 10 + i)
            for failure_type in ['No_Failure', 'TWF', 'HDF', 'PWF']:
                if failure_type in df['failure_type'].unique():
                    subset = df[df['failure_type'] == failure_type][col]
                    if len(subset) > 0:
                        plt.hist(subset, alpha=0.5, label=failure_type, bins=20)
            plt.title(f'{col.split()[0]} by Failure Type')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/multi_output_data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Multi-output visualizations saved to 'visualizations/multi_output_data_overview.png'")

def save_processed_multi_output_data(X_train, X_test, y_binary_train, y_binary_test, 
                                    y_type_train, y_type_test, scaler, feature_columns, failure_type_map):
    """
    Save processed multi-output data for later use
    """
    print("\n💾 Saving processed multi-output data...")
    
    # Create data directory
    os.makedirs('processed_data', exist_ok=True)
    
    # Save arrays
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_binary_train.npy', y_binary_train)
    np.save('processed_data/y_binary_test.npy', y_binary_test)
    np.save('processed_data/y_type_train.npy', y_type_train)
    np.save('processed_data/y_type_test.npy', y_type_test)
    
    # Save feature columns
    with open('processed_data/feature_columns.txt', 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    # Save failure type mapping
    import json
    with open('processed_data/failure_type_map.json', 'w') as f:
        json.dump(failure_type_map, f, indent=2)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'processed_data/scaler.pkl')
    
    print("✅ Processed multi-output data saved to 'processed_data/' directory")
    print(f"   - X_train.npy: {X_train.shape}")
    print(f"   - X_test.npy: {X_test.shape}")
    print(f"   - y_binary_train.npy: {y_binary_train.shape}")
    print(f"   - y_binary_test.npy: {y_binary_test.shape}")
    print(f"   - y_type_train.npy: {y_type_train.shape}")
    print(f"   - y_type_test.npy: {y_type_test.shape}")
    print(f"   - scaler.pkl")
    print(f"   - feature_columns.txt")
    print(f"   - failure_type_map.json")

def main():
    """
    Main function to execute Enhanced Phase 1A
    """
    print("=" * 80)
    print("🚀 PHASE 1A ENHANCED: MULTI-OUTPUT DATA PREPARATION")
    print("   AI4I 2020 Predictive Maintenance Dataset")
    print("   Binary Classification: Machine Failure (Yes/No)")
    print("   Multi-class Classification: Failure Type (TWF/HDF/PWF/OSF/RNF)")
    print("=" * 80)
    
    # Step 1: Download dataset (reuse existing if available)
    cached_path = "/home/codespace/.cache/kagglehub/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/versions/2"
    if os.path.exists(cached_path):
        print(f"🔄 Using cached dataset: {cached_path}")
        dataset_path = cached_path
    else:
        dataset_path = download_dataset()
        if not dataset_path:
            return
    
    # Step 2: Load and explore data
    df = load_and_explore_data(dataset_path)
    if df is None:
        return
    
    # Step 3: Preprocess data for multi-output
    X, y_binary, y_failure_type, failure_type_map = preprocess_data_multi_output(df)
    if X is None:
        return
    
    # Step 4: Apply advanced balancing
    X_balanced, y_binary_balanced, y_type_balanced = apply_advanced_balancing(X, y_binary, y_failure_type)
    
    # Step 5: Create train-test split
    X_train, X_test, y_binary_train, y_binary_test, y_type_train, y_type_test, scaler = \
        create_train_test_split_multi_output(X_balanced, y_binary_balanced, y_type_balanced)
    
    # Step 6: Visualize data
    visualize_multi_output_data(df, X, y_binary, y_failure_type, failure_type_map)
    
    # Step 7: Save processed data
    save_processed_multi_output_data(X_train, X_test, y_binary_train, y_binary_test, 
                                    y_type_train, y_type_test, scaler, X.columns, failure_type_map)
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1A ENHANCED COMPLETED SUCCESSFULLY!")
    print(f"🎯 Multi-Output Dataset Ready:")
    print(f"   📊 Training samples: {X_train.shape[0]:,}")
    print(f"   📊 Test samples: {X_test.shape[0]:,}")
    print(f"   📊 Features: {X_train.shape[1]}")
    print(f"   🔍 Binary target: Machine Failure (0/1)")
    print(f"   🔍 Multi-class target: Failure Type (6 classes)")
    print(f"   ⚖️ Data balanced: ✅")
    print(f"   📏 Data normalized: ✅")
    print("=" * 80)

if __name__ == "__main__":
    main()
