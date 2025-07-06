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


def test_elasticnet_full():
    """Test with full dataset."""
    print("\n=== Full Dataset Test ===")
    
    # Load full datasets
    X_train_shape, X_train_data = load_npy_as_array('X_train.npy')
    y_train_shape, y_train_data = load_npy_as_array('y_train.npy')
    X_test_shape, X_test_data = load_npy_as_array('X_test.npy')
    y_test_shape, y_test_data = load_npy_as_array('y_test.npy')
    
    print('feature mean', sum(X_train_data) / len(X_train_data))

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
    model.train(X_train_data, y_train_data, 2000, 1e-6)
    
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


def main():
    """Main test function."""
    print("ElasticNet MicroPython Module Test")
    print("==================================")
    
    try:
        # Test with full dataset
        model2 = test_elasticnet_full()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import sys
        sys.print_exception(e)

if __name__ == "__main__":
    main()
