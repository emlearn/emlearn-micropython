
#!/usr/bin/env python3
"""
Download and preprocess California housing dataset for MicroPython testing.
Saves scaled train/test splits as .npy files.
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import time


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_california_housing_data(sample=None):
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



def load_data():
    """Load the preprocessed California housing data."""
    print("Loading data...")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    print(f"Data types: X={X_train.dtype}, y={y_train.dtype}")
    
    return X_train, X_test, y_train, y_test

def test_elasticnet_configurations():
    """Test different ElasticNet configurations to find good baselines."""
    
    X_train, X_test, y_train, y_test = load_data()
    
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

def detailed_analysis_best_model(results):
    """Perform detailed analysis on the best performing model."""
    
    # Find best model by test MSE
    best_result = min(results, key=lambda x: x['test_mse'])
    print(f"\n" + "="*50)
    print("Detailed Analysis - Best Model")
    print("="*50)
    print(f"Best configuration: {best_result['config']}")
    print(f"Alpha: {best_result['alpha']}, L1 ratio: {best_result['l1_ratio']}")
    print(f"Test MSE: {best_result['test_mse']:.6f}")
    print(f"Test RMSE: {np.sqrt(best_result['test_mse']):.6f}")
    print(f"Test R²: {best_result['r2']:.6f}")
    
    model = best_result['model']
    
    # Load data again for detailed analysis
    X_train, X_test, y_train, y_test = load_data()
    
    # Show coefficients (if available)
    if hasattr(model, 'coef_'):
        print(f"\nModel coefficients:")
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
            print(f"  {name:12}: {coef:8.4f}")
        
        if hasattr(model, 'intercept_'):
            print(f"  {'Intercept':12}: {model.intercept_:8.4f}")
        
        # Count non-zero coefficients
        non_zero = np.sum(np.abs(model.coef_) > 1e-6)
        print(f"\nSparsity: {non_zero}/{len(model.coef_)} non-zero coefficients")
    
    # Sample predictions
    print(f"\nSample predictions (first 10 test samples):")
    y_test_pred = model.predict(X_test)
    print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-"*30)
    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        predicted = y_test_pred[i]
        error = abs(actual - predicted)
        print(f"{actual:<10.3f} {predicted:<10.3f} {error:<10.3f}")
    
    return best_result

def compare_with_micropython_format():
    """Create a comparison that matches the MicroPython module format."""
    
    print(f"\n" + "="*50)
    print("MicroPython Module Comparison Format")
    print("="*50)
    
    X_train, X_test, y_train, y_test = load_data()
    
    # Test with parameters similar to what MicroPython module might use
    configs_mp = [
        (0.01, 0.5, "emlearn_linreg equivalent 1"),
        (0.001, 0.5, "emlearn_linreg equivalent 2"),
        (0.1, 0.5, "emlearn_linreg equivalent 3"),
    ]
    
    for alpha, l1_ratio, description in configs_mp:
        print(f"\nTesting: {description}")
        print(f"Parameters: alpha={alpha}, l1_ratio={l1_ratio}")
        
        # Train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        
        # Test on small subset (like MicroPython test)
        n_small = 100
        X_small = X_train[:n_small]
        y_small = y_train[:n_small]
        
        small_mse = mean_squared_error(y_small, model.predict(X_small))
        full_train_mse = mean_squared_error(y_train, model.predict(X_train))
        test_mse = mean_squared_error(y_test, model.predict(X_test))
        
        print(f"  Small subset MSE (100 samples): {small_mse:.6f}")
        print(f"  Full training MSE: {full_train_mse:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        
        # Show first sample prediction for debugging
        first_pred = model.predict(X_test[:1])[0]
        first_actual = y_test[0]
        print(f"  First test sample: actual={first_actual:.3f}, predicted={first_pred:.3f}")
        
        # Show learned parameters
        print(f"  Learned bias: {model.intercept_:.6f}")
        print(f"  Weight range: [{model.coef_.min():.6f}, {model.coef_.max():.6f}]")
        print(f"  Non-zero weights: {np.sum(np.abs(model.coef_) > 1e-6)}/{len(model.coef_)}")

def create_reference_outputs():
    """Create reference outputs for validating MicroPython implementation."""
    
    print(f"\n" + "="*50)
    print("Reference Outputs for MicroPython Validation")
    print("="*50)
    
    X_train, X_test, y_train, y_test = load_data()
    
    # Use specific parameters for reference
    alpha, l1_ratio = 0.01, 0.5
    
    print(f"Reference model: alpha={alpha}, l1_ratio={l1_ratio}")
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    
    # Save reference results
    reference_data = {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'intercept': model.intercept_,
        'coefficients': model.coef_,
        'train_mse': mean_squared_error(y_train, model.predict(X_train)),
        'test_mse': mean_squared_error(y_test, model.predict(X_test)),
        'first_test_prediction': model.predict(X_test[:1])[0],
        'first_test_actual': y_test[0]
    }
    
    print(f"Intercept: {reference_data['intercept']:.8f}")
    print(f"Coefficients: {reference_data['coefficients']}")
    print(f"Training MSE: {reference_data['train_mse']:.8f}")
    print(f"Test MSE: {reference_data['test_mse']:.8f}")
    print(f"First test prediction: {reference_data['first_test_prediction']:.8f}")
    print(f"First test actual: {reference_data['first_test_actual']:.8f}")
    
    # Save to file for MicroPython comparison
    np.savez('reference_results.npz', **reference_data)
    print(f"\nReference results saved to 'reference_results.npz'")
    
    return reference_data


def main():

    # Prepare the data
    prepare_california_housing_data(sample=4000)

    # Test different configurations
    results = test_elasticnet_configurations()
    
    # Detailed analysis of best model
    best_result = detailed_analysis_best_model(results)
    
    # Compare with MicroPython format
    compare_with_micropython_format()
    
    # Create reference outputs
    reference_data = create_reference_outputs()
        
    print(f"\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Best overall performance: {best_result['config']}")
    print(f"Best test MSE: {best_result['test_mse']:.6f}")
    print(f"Target for MicroPython module: MSE < {best_result['test_mse']:.3f}")
    print("\nFiles created:")
    print("- reference_results.npz (for MicroPython validation)")


if __name__ == "__main__":
    main()
 
