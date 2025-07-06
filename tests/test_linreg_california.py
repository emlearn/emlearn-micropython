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


def train_with_monitoring(model, X_train, y_train, max_iterations=1000, 
                         tolerance=1e-6, check_interval=10, verbose=True):
    """
    Train model with progress monitoring using generator pattern.
    
    Yields: (iteration, mse, change, converged, diverged)
    """
    prev_mse = float('inf')
    n_features = model.get_n_features()
    
    for iteration in range(max_iterations):
        # Perform one gradient descent step
        model.step(X_train, y_train)
        
        # Check progress at intervals
        if iteration % check_interval == 0:
            # Calculate current MSE
            current_mse = model.score_mse(X_train, y_train)
            change = abs(prev_mse - current_mse)
            
            # Check convergence
            converged = change < tolerance and iteration > check_interval * 2
            
            # Check divergence
            diverged = current_mse > prev_mse * 10.0 or not (current_mse == current_mse)  # NaN check
            
            # Yield progress
            yield iteration, current_mse, change, converged, diverged
            
            if converged:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            if diverged:
                if verbose:
                    print(f"Diverged at iteration {iteration}")
                break
                
            prev_mse = current_mse
        else:
            # Yield minimal info for non-check iterations
            yield iteration, None, None, False, False

def train_model(model, X_train, y_train, max_iterations=1000, tolerance=1e-6, 
                check_interval=10, verbose=True):
    """
    Simple training function with progress monitoring.
    """
    if verbose:
        print(f"Training model for up to {max_iterations} iterations...")
        print(f"Checking convergence every {check_interval} iterations")
        print("Iter    MSE       Change    Status")
        print("-" * 40)
    
    for iter_num, mse, change, converged, diverged in train_with_monitoring(
        model, X_train, y_train, max_iterations, tolerance, check_interval, verbose=False
    ):
        
        if verbose and mse is not None:  # Only print on check intervals
            status = ""
            if converged:
                status = "CONVERGED"
            elif diverged:
                status = "DIVERGED"
            
            print(f"{iter_num:4d}    {mse:8.6f}  {change:8.6f}  {status}")
        
        if converged or diverged:
            break
    
    if verbose:
        print("Training completed.\n")


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
        train_model(model, X_subset, y_subset, max_iterations=300, check_interval=50, verbose=False)
        
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
