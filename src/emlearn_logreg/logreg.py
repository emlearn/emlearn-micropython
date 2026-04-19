import array

log_prefix = 'emlearn_logreg:'


def _make_workspace_triplet(n_classes):
    return (
        array.array('f', [0.0] * n_classes),
        array.array('f', [0.0] * n_classes),
        array.array('f', [0.0] * n_classes),
    )


def _make_workspace_pair(n_classes):
    return (
        array.array('f', [0.0] * n_classes),
        array.array('f', [0.0] * n_classes),
    )


def _make_predict_buffers(n_classes):
    return (
        array.array('f', [0.0] * n_classes),
        array.array('f', [0.0] * n_classes),
    )

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
    n_classes = model.get_n_classes()
    if len(X_train) % n_features != 0:
        raise ValueError('X_train size mismatch with n_features')
    n_samples = len(X_train) // n_features

    if len(y_train) != n_samples * n_classes:
        raise ValueError('y_train must be one-hot encoded (len = n_samples * n_classes)')
    if n_samples == 0:
        raise ValueError('y_train is empty')

    if batch_size is None or batch_size <= 0 or batch_size > n_samples:
        batch_size = n_samples

    logits_buf, probs_buf, bias_buf = _make_workspace_triplet(n_classes)
    score_logits, score_probs = _make_workspace_pair(n_classes)
    predict_logits, predict_probs = _make_predict_buffers(n_classes)

    prev_loss = float('inf')

    use_batches = batch_size < n_samples
    if use_batches:
        batch_X = array.array('f', [0.0] * (batch_size * n_features))
        batch_y = array.array('f', [0.0] * (batch_size * n_classes))
        batch_X_view = memoryview(batch_X)
        batch_y_view = memoryview(batch_y)
    else:
        batch_X_view = memoryview(X_train)
        batch_y_view = memoryview(y_train)

    for iteration in range(max_iterations):
        if use_batches:
            for start in range(0, n_samples, batch_size):
                count = min(batch_size, n_samples - start)
                base_feature = start * n_features
                base_target = start * n_classes
                # Copy features for current batch
                end_f = base_feature + count * n_features
                batch_X[:count * n_features] = X_train[base_feature:end_f]
                # Copy targets
                end_t = base_target + count * n_classes
                batch_y[:count * n_classes] = y_train[base_target:end_t]

                X_slice = batch_X_view[:count * n_features]
                y_slice = batch_y_view[:count * n_classes]
                model.step(X_slice, y_slice, logits_buf, probs_buf, bias_buf)
        else:
            model.step(batch_X_view, batch_y_view, logits_buf, probs_buf, bias_buf)

        if iteration % check_interval != 0:
            continue

        current_loss = model.score_logloss(batch_X_view, batch_y_view, score_logits, score_probs)
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

