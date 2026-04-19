
#!/usr/bin/env python3
"""
Download and preprocess California housing dataset for MicroPython testing.
Saves scaled train/test splits as .npy files.
"""

import os
import time

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_california_housing_data(data_dir, sample=None):
    """Download, preprocess and save California housing dataset."""
    
    print("Downloading California housing dataset...")
    # Load the dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    if sample is not None:
        indices = np.random.choice(X.shape[0], size=sample, replace=False)
        X = X[indices]
        y = y[indices]

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
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
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



def load_data(data_dir):
    """Load the preprocessed California housing data."""
    print("Loading data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    print(f"Data types: X={X_train.dtype}, y={y_train.dtype}")
    
    return X_train, X_test, y_train, y_test

def test_elasticnet_configurations(data_dir):
    """Test different ElasticNet configurations to find good baselines."""
    
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    # Test configurations: (alpha, l1_ratio, description)
    configs = [
        (0.0, 0.0, "No regularization (OLS)"),
        (0.01, 0.0, "Ridge (alpha=0.01)"),
        (0.01, 1.0, "LASSO (alpha=0.01)"),
        (0.01, 0.5, "ElasticNet (alpha=0.01, l1_ratio=0.5)"),
        (0.001, 0.5, "ElasticNet (alpha=0.001, l1_ratio=0.5)"),
        (0.1, 0.5, "ElasticNet (alpha=0.1, l1_ratio=0.5)"),
    ]
    
    print("\n" + "="*70)
    print("ElasticNet Configuration Comparison")
    print("="*70)
    print(f"{'Configuration':<35} {'Train MSE':<12} {'Test MSE':<12} {'R²':<8} {'Time':<8}")
    print("-"*70)
    
    results = []
    
    for alpha, l1_ratio, description in configs:
        start_time = time.time()
        
        # Create and train model
        if alpha == 0.0:
            # Use regular linear regression for no regularization
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        elapsed_time = time.time() - start_time
        
        print(f"{description:<35} {train_mse:<12.6f} {test_mse:<12.6f} {test_r2:<8.3f} {elapsed_time:<8.3f}")
        
        results.append({
            'config': description,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'r2': test_r2,
            'time': elapsed_time,
            'model': model
        })
    
    return results



def main():

    here = os.path.dirname(__file__)
    data_dir = here

    # Prepare the data
    prepare_california_housing_data(data_dir, sample=4000)

    # Test different configurations
    results = test_elasticnet_configurations(data_dir)

    #print(results)


if __name__ == "__main__":
    main()
 
