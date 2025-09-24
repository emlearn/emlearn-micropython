
#!/usr/bin/env python3
"""
Download and preprocess California housing dataset for MicroPython testing.
Saves scaled train/test splits as .npy files.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_california_housing_data():
    """Download, preprocess and save California housing dataset."""
    
    print("Downloading California housing dataset...")
    # Load the dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {housing.feature_names}")
    print(f"Target: median house value in hundreds of thousands of dollars")
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Scale the features (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nScaling applied:")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature stds: {scaler.scale_}")
    
    # Convert to float32 for MicroPython compatibility
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Save as .npy files
    np.save('X_train.npy', X_train_scaled)
    np.save('X_test.npy', X_test_scaled)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    print("\nSaved files:")
    print(f"X_train.npy: {X_train_scaled.shape} float32")
    print(f"X_test.npy: {X_test_scaled.shape} float32")
    print(f"y_train.npy: {y_train.shape} float32")
    print(f"y_test.npy: {y_test.shape} float32")
    
    # Print some statistics for verification
    print("\nData statistics:")
    print(f"X_train range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    print(f"y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"y_train mean: {y_train.mean():.3f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test



if __name__ == "__main__":
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_california_housing_data()
    
    print("\nData preparation complete!")
    print("Files ready for MicroPython testing:")
    print("- X_train.npy, X_test.npy, y_train.npy, y_test.npy")
