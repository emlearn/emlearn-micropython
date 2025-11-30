"""
Training helper for EML PLS Regression MicroPython module
"""

log_prefix = 'emlearn_plsr:'

def fit(model, X_train, y_train,
        max_iterations=100, 
        tolerance=1e-6,
        check_interval=10,
        verbose=0,
        ):
    """
    Simple training loop for PLSR
    
    Args:
        model: PLSR model instance
        X_train: Training input data [n_samples x n_features], float32 array
        y_train: Training target data [n_samples], float32 array
        max_iterations: Maximum iterations per component (default: 100)
        tolerance: Convergence tolerance (default: 1e-6)
        check_interval: Check convergence every N iterations (default: 10)
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed)
    
    Returns:
        tuple: (total_iterations, final_convergence_metric)
    """
    
    # Start training
    model.fit_start(X_train, y_train)
    
    total_iterations = 0
    component = 0
    
    # Train all components
    while not model.is_complete():
        
        # Iterate until convergence for current component
        component_iterations = 0
        
        while not model.is_converged() and component_iterations < max_iterations:
            # Perform iteration step
            model.step(tolerance)
            component_iterations += 1
            total_iterations += 1
            
            # Check and report progress at intervals
            if verbose >= 2 and component_iterations % check_interval == 0:
                metric = model.get_convergence_metric()
                print(log_prefix, f'  Iteration {component_iterations}: convergence={metric:.6e}')
        
        # Check if converged
        if not model.is_converged():
            if verbose >= 1:
                print(log_prefix, f'  WARNING: Component N did not converge after {component_iterations} iterations')
            break
        
        if verbose >= 1:
            metric = model.get_convergence_metric()
            print(log_prefix, f'  Converged after {component_iterations} iterations (metric={metric:.6e})')
        
        # Finalize component
        model.finalize_component()
        component += 1
    
    final_metric = model.get_convergence_metric()
    
    if verbose >= 1:
        if model.is_complete():
            print(log_prefix, f'Training complete: {total_iterations} total iterations')
        else:
            print(log_prefix, f'Training incomplete after {total_iterations} iterations')
    
    return total_iterations, final_metric


