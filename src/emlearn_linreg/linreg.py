

log_prefix = 'emlearn_linreg:'

def train(model, X_train, y_train,
        max_iterations=100, 
        tolerance=1e-6,
        check_interval=10,
        divergence_factor=10.0,
        score_limit=None,
        verbose=0,
        ):
    """
    Simple training loop
    """
    prev_mse = float('inf')
    
    for iteration in range(max_iterations):
        # Perform one gradient descent step
        model.step(X_train, y_train)

        # Only check progress at intervals        
        if iteration % check_interval != 0:
            continue

        # Calculate current MSE
        current_mse = model.score_mse(X_train, y_train)
        change = abs(prev_mse - current_mse)

        if verbose >= 2:
            print(log_prefix, f'Iteration {iteration} mse={current_mse}')        

        # Check convergence
        converged = change < tolerance and iteration > check_interval * 2
        
        if score_limit is not None:
            converged = converged or current_mse <= score_limit

        # Check divergence
        diverged = current_mse > prev_mse * divergence_factor or not (current_mse == current_mse)  # NaN check
        
        if converged:
            if verbose >= 1:
                print(log_prefix, f"Converged at iteration {iteration}")
            break
            
        if diverged:
            if verbose >= 1:
                print(log_prefix, f"Diverged at iteration {iteration}")
            break
            
        prev_mse = current_mse

    return iteration, prev_mse

