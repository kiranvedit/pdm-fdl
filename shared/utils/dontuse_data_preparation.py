#!/usr/bin/env python3
"""
Phase 1A: Data Preparation Script
Download and preprocess the AI4I 2020 Predictive Maintenance Dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
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
    
    # Check target distribution
    if 'Machine failure' in df.columns:
        target_dist = df['Machine failure'].value_counts()
        print(f"\n🎯 Target Distribution (Machine failure):")
        print(target_dist)
        print(f"📊 Class Balance Ratio: {target_dist.iloc[1] / target_dist.iloc[0]:.3f}")
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    """
    print("\n🔧 Preprocessing data...")
    
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
    
    # Separate features and target
    if 'Machine failure' in processed_df.columns:
        X = processed_df.drop('Machine failure', axis=1)
        y = processed_df['Machine failure']
        
        # Also remove specific failure types if they exist
        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        existing_failure_cols = [col for col in failure_columns if col in X.columns]
        if existing_failure_cols:
            X = X.drop(existing_failure_cols, axis=1)
            print(f"🗑️ Removed specific failure columns: {existing_failure_cols}")
        
        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        print(f"📋 Feature columns: {list(X.columns)}")
        
        return X, y
    else:
        print("❌ Target column 'Machine failure' not found")
        return None, None

def apply_smote_balancing(X, y):
    """
    Apply SMOTE to balance the dataset
    """
    print("\n⚖️ Applying SMOTE for class balancing...")
    
    # Check original distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 Original distribution: {dict(zip(unique, counts))}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Check new distribution
    unique, counts = np.unique(y_balanced, return_counts=True)
    print(f"📊 Balanced distribution: {dict(zip(unique, counts))}")
    print(f"✅ New dataset shape: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train-test split for the data
    """
    print(f"\n✂️ Creating train-test split ({int((1-test_size)*100)}% / {int(test_size*100)}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📈 Training set: {X_train.shape}")
    print(f"📈 Test set: {X_test.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features normalized using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def visualize_data(df, X, y):
    """
    Create visualizations for the dataset
    """
    print("\n📊 Creating data visualizations...")
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Target distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    y.value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Target Distribution\n(Machine Failure)')
    plt.xlabel('Machine Failure')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # 2. Feature correlations
    plt.subplot(2, 3, 2)
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix.iloc[:5, :5], annot=True, cmap='coolwarm', 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlations\n(Top 5x5)')
    
    # 3. Distribution of key features
    if 'Air temperature [K]' in df.columns:
        plt.subplot(2, 3, 3)
        df['Air temperature [K]'].hist(bins=30, alpha=0.7, color='green')
        plt.title('Air Temperature Distribution')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Frequency')
    
    if 'Rotational speed [rpm]' in df.columns:
        plt.subplot(2, 3, 4)
        df['Rotational speed [rpm]'].hist(bins=30, alpha=0.7, color='orange')
        plt.title('Rotational Speed Distribution')
        plt.xlabel('Speed (RPM)')
        plt.ylabel('Frequency')
    
    if 'Torque [Nm]' in df.columns:
        plt.subplot(2, 3, 5)
        df['Torque [Nm]'].hist(bins=30, alpha=0.7, color='purple')
        plt.title('Torque Distribution')
        plt.xlabel('Torque (Nm)')
        plt.ylabel('Frequency')
    
    if 'Tool wear [min]' in df.columns:
        plt.subplot(2, 3, 6)
        df['Tool wear [min]'].hist(bins=30, alpha=0.7, color='red')
        plt.title('Tool Wear Distribution')
        plt.xlabel('Tool Wear (min)')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('visualizations/data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Visualizations saved to 'visualizations/data_overview.png'")

def save_processed_data(X_train, X_test, y_train, y_test, scaler, feature_columns):
    """
    Save processed data for later use
    """
    print("\n💾 Saving processed data...")
    
    # Create data directory
    os.makedirs('processed_data', exist_ok=True)
    
    # Save arrays
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/y_test.npy', y_test)
    
    # Save feature columns
    with open('processed_data/feature_columns.txt', 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'processed_data/scaler.pkl')
    
    print("✅ Processed data saved to 'processed_data/' directory")
    print(f"   - X_train.npy: {X_train.shape}")
    print(f"   - X_test.npy: {X_test.shape}")
    print(f"   - y_train.npy: {y_train.shape}")
    print(f"   - y_test.npy: {y_test.shape}")
    print(f"   - scaler.pkl")
    print(f"   - feature_columns.txt")

def main():
    """
    Main function to execute Phase 1A
    """
    print("=" * 80)
    print("🚀 PHASE 1A: DATA PREPARATION")
    print("   AI4I 2020 Predictive Maintenance Dataset")
    print("=" * 80)
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        return
    
    # Step 2: Load and explore data
    df = load_and_explore_data(dataset_path)
    if df is None:
        return
    
    # Step 3: Preprocess data
    X, y = preprocess_data(df)
    if X is None or y is None:
        return
    
    # Step 4: Apply SMOTE balancing
    X_balanced, y_balanced = apply_smote_balancing(X, y)
    
    # Step 5: Create train-test split
    X_train, X_test, y_train, y_test, scaler = create_train_test_split(X_balanced, y_balanced)
    
    # Step 6: Visualize data
    visualize_data(df, X, y)
    
    # Step 7: Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, scaler, X.columns)
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1A COMPLETED SUCCESSFULLY!")
    print(f"📊 Final dataset ready for model training:")
    print(f"   - Training samples: {X_train.shape[0]:,}")
    print(f"   - Test samples: {X_test.shape[0]:,}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Classes balanced: ✅")
    print(f"   - Data normalized: ✅")
    print("=" * 80)

if __name__ == "__main__":
    main()
