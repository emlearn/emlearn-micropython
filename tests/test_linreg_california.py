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
            max_iterations=2000, check_interval=50, verbose=True)

        # Calculate predictions and MSE
        mse = model.score_mse(X_subset, y_subset)
        
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
        # Compare regularization approaches
        compare_regularization()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import sys
        sys.print_exception(e)

if __name__ == "__main__":
    main()
