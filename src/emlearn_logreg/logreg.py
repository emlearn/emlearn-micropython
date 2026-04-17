

import array

log_prefix = 'emlearn_logreg:'

def train(model, X_train, y_train,
        max_iterations=200,
        tolerance=1e-4,
        check_interval=5,
        divergence_factor=2.0,
        score_limit=None,
        verbose=0,
        batch_size=None,
        ):
    """Mini-batch training loop for logistic regression.

    Copies data into a reusable buffer when mini-batching to limit peak memory.
    """
    n_features = model.get_n_features()
    n_samples = len(y_train)
    if n_samples == 0:
        raise ValueError('y_train is empty')

    if len(X_train) != n_samples * n_features:
        raise ValueError('X_train size mismatch with y_train and n_features')

    if batch_size is None or batch_size <= 0 or batch_size > n_samples:
        batch_size = n_samples

    prev_loss = float('inf')

    use_batches = batch_size < n_samples
    if use_batches:
        batch_X = array.array('f', [0.0] * (batch_size * n_features))
        batch_y = array.array('f', [0.0] * batch_size)
        batch_X_view = memoryview(batch_X)
        batch_y_view = memoryview(batch_y)

    for iteration in range(max_iterations):
        if use_batches:
            for start in range(0, n_samples, batch_size):
                count = min(batch_size, n_samples - start)
                base_feature = start * n_features
                # Copy features for current batch
                for idx in range(count * n_features):
                    batch_X[idx] = X_train[base_feature + idx]
                # Copy targets
                for idx in range(count):
                    batch_y[idx] = y_train[start + idx]

                X_slice = batch_X_view[:count * n_features]
                y_slice = batch_y_view[:count]
                model.step(X_slice, y_slice)
        else:
            model.step(X_train, y_train)

        if iteration % check_interval != 0:
            continue

        current_loss = model.score_logloss(X_train, y_train)
        change = abs(prev_loss - current_loss)

        if verbose >= 2:
            print(log_prefix, f'Iteration {iteration} loss={current_loss}')

        converged = change < tolerance and iteration > check_interval * 2

        if score_limit is not None:
            converged = converged or current_loss <= score_limit

        diverged = (current_loss > prev_loss * divergence_factor) or not (current_loss == current_loss)

        if converged:
            if verbose >= 1:
                print(log_prefix, f"Converged at iteration {iteration}")
            break

        if diverged:
            if verbose >= 1:
                print(log_prefix, f"Diverged at iteration {iteration}")
            break

        prev_loss = current_loss

    return iteration, prev_loss

