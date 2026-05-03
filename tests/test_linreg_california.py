
import emlearn_linreg
import npyfile
import array
import gc
import time

data_dir = './examples/datasets/california/'

def load_npy(filename):
    shape, data = npyfile.load(filename)
    return shape, data

def test_elasticnet_full():
    """Test with full dataset."""
    # Load full datasets
    X_train_shape, X_train_data = load_npy(data_dir+'X_train.npy')
    y_train_shape, y_train_data = load_npy(data_dir+'y_train.npy')
    X_test_shape, X_test_data = load_npy(data_dir+'X_test.npy')
    y_test_shape, y_test_data = load_npy(data_dir+'y_test.npy')
    
    n_features = X_train_shape[1]
    n_train = y_train_shape[0]
    n_test = y_test_shape[0]
    
    print(f"Train set: {n_train} samples")
    print(f"Test set: {n_test} samples")
    print(f"Features: {n_features}")
    
    # Create model with different hyperparameters for full dataset
    # Lower learning rate for stability
    model = emlearn_linreg.new(n_features, 0.001, 0.5, 0.1)  
    
    train_start = time.ticks_ms()
    # Train on full dataset
    print("Training on full dataset...")
    stop_iter, stop_mse = emlearn_linreg.train(model,
            X_train_data, y_train_data,
            max_iterations=100, check_interval=10,
            verbose=2, tolerance=0.0001, score_limit=0.55,
    )
    train_duration = time.ticks_diff(time.ticks_ms(), train_start)    
    print('Train time (ms)', train_duration, 'per iter', train_duration/stop_iter)
    
    # Evaluate
    train_mse = model.score_mse(X_train_data, y_train_data)
    test_mse = model.score_mse(X_test_data, y_test_data)

    assert train_mse <= 0.65, train_mse
    assert test_mse <= 0.65, test_mse
      

