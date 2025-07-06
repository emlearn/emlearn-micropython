# MicroPython test script for ElasticNet module with California housing dataset
import emlearn_linreg
import npyfile
import array
import gc

def load_npy_as_array(filename, dtype='f'):
    """Load .npy file and convert to MicroPython array."""
    print(f"Loading {filename}...")
    shape, data = npyfile.load(filename)
    print(f"Shape: {shape}, Data type: {type(data)}")
    
    # Convert to flat array
    if isinstance(data, (list, tuple)):
        # If data is nested, flatten it
        flat_data = []
        def flatten(lst):
            for item in lst:
                if isinstance(item, (list, tuple)):
                    flatten(item)
                else:
                    flat_data.append(item)
        flatten(data)
        return shape, array.array(dtype, flat_data)
    else:
        # If data is already flat
        return shape, array.array(dtype, data)

def test_elasticnet_small():
    """Test with a small subset of data first."""
    print("=== Small Dataset Test ===")
    
    # Load small subset for initial testing
    X_shape, X_data = load_npy_as_array('X_train.npy')
    y_shape, y_data = load_npy_as_array('y_train.npy')
    
    n_features = X_shape[1]  # Should be 8
    n_samples_small = min(100, y_shape[0])  # Use first 100 samples
    
    print(f"Using {n_samples_small} samples with {n_features} features")
    
    # Extract subset
    X_small = array.array('f', X_data[:n_samples_small * n_features])
    y_small = array.array('f', y_data[:n_samples_small])
    
    # Create model
    print("Creating ElasticNet model...")
    model = emlearn_linreg.new(n_features, 0.01, 0.5, 0.001)
    
    # Train
    print("Training model...")
    model.train(X_small, y_small, 500, 1e-6)
    
    # Get parameters
    weights = array.array('f', [0.0] * n_features)
    model.get_weights(weights)
    bias = model.get_bias()
    
    print(f"Learned weights: {list(weights)}")
    print(f"Learned bias: {bias}")
    
    # Test prediction
    test_features = array.array('f', X_data[:n_features])  # First sample
    prediction = model.predict(test_features)
    actual = y_data[0]
    
    print(f"First sample prediction: {prediction}")
    print(f"First sample actual: {actual}")
    print(f"Error: {abs(prediction - actual)}")
    
    # Test prediction and calculate MSE manually
    y_pred = array.array('f', [0.0] * n_samples_small)
    for i in range(n_samples_small):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        sample_features = array.array('f', X_data[start_idx:end_idx])
        prediction = model.predict(sample_features)
        y_pred[i] = prediction
    
    # Calculate training MSE using the cleaner API
    mse = model.mse(y_small, y_pred)
    print(f"Training MSE: {mse}")
    
    return model

def test_elasticnet_full():
    """Test with full dataset."""
    print("\n=== Full Dataset Test ===")
    
    # Load full datasets
    X_train_shape, X_train_data = load_npy_as_array('X_train.npy')
    y_train_shape, y_train_data = load_npy_as_array('y_train.npy')
    X_test_shape, X_test_data = load_npy_as_array('X_test.npy')
    y_test_shape, y_test_data = load_npy_as_array('y_test.npy')
    
    n_features = X_train_shape[1]
    n_train = y_train_shape[0]
    n_test = y_test_shape[0]
    
    print(f"Train set: {n_train} samples")
    print(f"Test set: {n_test} samples")
    print(f"Features: {n_features}")
    
    # Create model with different hyperparameters for full dataset
    print("Creating ElasticNet model...")
    model = emlearn_linreg.new(n_features, 0.001, 0.5, 0.0001)  # Lower learning rate for stability
    
    # Train on full dataset
    print("Training on full dataset...")
    model.train(X_train_data, y_train_data, 1000, 1e-6)
    
    # Get final parameters
    weights = array.array('f', [0.0] * n_features)
    model.get_weights(weights)
    bias = model.get_bias()
    
    print(f"Final weights: {list(weights)}")
    print(f"Final bias: {bias}")
    
    # Evaluate on train set
    print("Calculating training MSE...")
    y_train_pred = array.array('f', [0.0] * n_train)
    for i in range(n_train):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        sample_features = array.array('f', X_train_data[start_idx:end_idx])
        y_train_pred[i] = model.predict(sample_features)
    
    train_mse = model.mse(y_train_data, y_train_pred)
    print(f"Training MSE: {train_mse}")
    
    # Evaluate on test set
    print("Calculating test MSE...")
    y_test_pred = array.array('f', [0.0] * n_test)
    for i in range(n_test):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        sample_features = array.array('f', X_test_data[start_idx:end_idx])
        y_test_pred[i] = model.predict(sample_features)
    
    test_mse = model.mse(y_test_data, y_test_pred)
    print(f"Test MSE: {test_mse}")
    
    # Make some sample predictions
    print("\nSample predictions:")
    for i in range(min(5, n_test)):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        test_features = array.array('f', X_test_data[start_idx:end_idx])
        prediction = model.predict(test_features)
        actual = y_test_data[i]
        print(f"Sample {i}: predicted={prediction:.3f}, actual={actual:.3f}, error={abs(prediction-actual):.3f}")
    
    return model

def compare_regularization():
    """Compare different regularization settings."""
    print("\n=== Regularization Comparison ===")
    
    # Load small dataset for quick comparison
    X_shape, X_data = load_npy_as_array('X_train.npy')
    y_shape, y_data = load_npy_as_array('y_train.npy')
    
    n_features = X_shape[1]
    n_samples = min(500, y_shape[0])
    
    X_subset = array.array('f', X_data[:n_samples * n_features])
    y_subset = array.array('f', y_data[:n_samples])
    
    configs = [
        ("No regularization", 0.0, 0.0),
        ("Ridge (L2)", 0.01, 0.0),
        ("LASSO (L1)", 0.01, 1.0),
        ("ElasticNet", 0.01, 0.5),
    ]
    
    for name, alpha, l1_ratio in configs:
        print(f"\nTesting {name} (alpha={alpha}, l1_ratio={l1_ratio}):")
        
        model = emlearn_linreg.new(n_features, alpha, l1_ratio, 0.001)
        model.train(X_subset, y_subset, 300, 1e-6)
        
        # Calculate predictions and MSE using cleaner API
        y_pred = array.array('f', [0.0] * n_samples)
        for i in range(n_samples):
            start_idx = i * n_features
            end_idx = start_idx + n_features
            sample_features = array.array('f', X_subset[start_idx:end_idx])
            y_pred[i] = model.predict(sample_features)
        
        mse = model.mse(y_subset, y_pred)
        
        weights = array.array('f', [0.0] * n_features)
        model.get_weights(weights)
        
        # Count non-zero weights (sparsity)
        non_zero = sum(1 for w in weights if abs(w) > 1e-6)
        
        print(f"  MSE: {mse:.6f}")
        print(f"  Non-zero weights: {non_zero}/{n_features}")
        print(f"  Max weight magnitude: {max(abs(w) for w in weights):.6f}")

def main():
    """Main test function."""
    print("ElasticNet MicroPython Module Test")
    print("==================================")
    
    try:
        # Test with small dataset first
        model1 = test_elasticnet_small()
        
        # Force garbage collection
        del model1
        gc.collect()
        
        # Test with full dataset
        model2 = test_elasticnet_full()
        
        # Force garbage collection
        del model2
        gc.collect()
        
        # Compare regularization approaches
        compare_regularization()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import sys
        sys.print_exception(e)

if __name__ == "__main__":
    main()
