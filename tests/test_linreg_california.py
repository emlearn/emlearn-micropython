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

    return shape, data


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
        
        model = emlearn_linreg.new(n_features, alpha, l1_ratio, 0.01)
        emlearn_linreg.train(model, X_subset, y_subset,
            max_iterations=2000, check_interval=50, verbose=1)

        # Calculate predictions and MSE
        mse = model.score_mse(X_subset, y_subset)
        
        weights = array.array('f', [0.0] * n_features)
        model.get_weights(weights)
        
        # Count non-zero weights (sparsity)
        non_zero = sum(1 for w in weights if abs(w) > 1e-6)
        print(f"  MSE: {mse:.6f}")
        print(f"  Non-zero weights: {non_zero}/{n_features}")
        print(f"  Max weight magnitude: {max(abs(w) for w in weights):.6f}")

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
    model = emlearn_linreg.new(n_features, 0.001, 0.5, 0.01)  # Lower learning rate for stability
    
    # Train on full dataset
    print("Training on full dataset...")
    emlearn_linreg.train(model, X_train_data, y_train_data,
        max_iterations=2000, check_interval=50, verbose=1, tolerance=0.001, score_limit=0.60)
    
    # Get final parameters
    weights = array.array('f', [0.0] * n_features)
    model.get_weights(weights)
    bias = model.get_bias()
    
    print(f"Final weights: {list(weights)}")
    print(f"Final bias: {bias}")
    
    # Evaluate on train set
    print("Calculating training MSE...")    
    train_mse = model.score_mse(X_train_data, y_train_data)
    print(f"Training MSE: {train_mse}")
    
    # Evaluate on test set
    print("Calculating test MSE...")
    test_mse = model.score_mse(X_test_data, y_test_data)
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

def main():
    """Main test function."""
    print("ElasticNet MicroPython Module Test")
    print("==================================")
    
    try:
        # Compare regularization approaches
        compare_regularization()
        
        test_elasticnet_full()        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import sys
        sys.print_exception(e)

if __name__ == "__main__":
    main()
