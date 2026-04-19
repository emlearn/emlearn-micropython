import array

# When used as external C module, the .py is the top-level import,
# and we need to merge the native module symbols at import time
# When used as dynamic native modules (.mpy), .py and native code is merged at build time
try:
    from emlearn_logreg_c import *
except ImportError:
    pass

log_prefix = 'emlearn_logreg:'


def _make_buffer(n):
    return array.array('f', [0.0] * n)

def train(model, X_train, y_train,
        max_iterations=200,
        tolerance=1e-4,
        check_interval=5,
        divergence_factor=2.0,
        score_limit=None,
        verbose=0,
        ):
    """Full-dataset training loop for logistic regression."""
    if max_iterations <= 0:
        raise ValueError('max_iterations must be positive')
    if check_interval <= 0:
        raise ValueError('check_interval must be positive')

    n_features = model.get_n_features()
    n_classes = model.get_n_classes()
    if len(X_train) % n_features != 0:
        raise ValueError('X_train size mismatch with n_features')
    n_samples = len(X_train) // n_features

    if len(y_train) != n_samples * n_classes:
        raise ValueError('y_train must be one-hot encoded (len = n_samples * n_classes)')
    if n_samples == 0:
        raise ValueError('y_train is empty')

    logits_buf = _make_buffer(n_classes)
    probs_buf = _make_buffer(n_classes)
    bias_buf = _make_buffer(n_classes)
    score_logits = _make_buffer(n_classes)
    score_probs = _make_buffer(n_classes)

    prev_loss = None
    final_loss = float('inf')
    iterations_completed = 0

    X_view = memoryview(X_train)
    y_view = memoryview(y_train)

    for _ in range(max_iterations):
        iterations_completed += 1
        model.step(X_view, y_view, logits_buf, probs_buf, bias_buf)

        if iterations_completed % check_interval != 0:
            continue

        current_loss = model.score_logloss(X_view, y_view, score_logits, score_probs)
        final_loss = current_loss
        change = float('inf') if prev_loss is None else abs(prev_loss - current_loss)

        if verbose >= 2:
            print(log_prefix, f'Iteration {iterations_completed} loss={current_loss}')

        converged = change < tolerance and iterations_completed > check_interval * 2

        if score_limit is not None and current_loss <= score_limit:
            converged = True

        diverged = not (current_loss == current_loss)
        if not diverged and prev_loss is not None:
            diverged = current_loss > prev_loss * divergence_factor

        if converged:
            if verbose >= 1:
                print(log_prefix, f"Converged at iteration {iterations_completed}")
            break

        if diverged:
            if verbose >= 1:
                print(log_prefix, f"Diverged at iteration {iterations_completed}")
            break

        prev_loss = current_loss

    if final_loss == float('inf'):
        final_loss = model.score_logloss(X_view, y_view, score_logits, score_probs)

    return iterations_completed, final_loss


def train_batches(model,
        batch_iter_factory,
        max_iterations=200,
        tolerance=1e-4,
        check_interval=5,
        divergence_factor=2.0,
        score_limit=None,
        verbose=0,
        score_batches=None,
        ):
    """Train logistic regression model using externally provided batches.

    batch_iter_factory must be a callable that returns a fresh iterator for each
    epoch. Each iterator should yield tuples of (X_batch, y_batch) where both are
    float32 arrays compatible with model.step(). y_batch must be one-hot encoded.

    score_batches is an optional callable taking the model and returning the
    average log-loss over the data (computed however the caller prefers). When
    provided, it is used for convergence checking.
    """
    if not callable(batch_iter_factory):
        raise ValueError('batch_iter_factory must be callable')
    if max_iterations <= 0:
        raise ValueError('max_iterations must be positive')
    if check_interval <= 0:
        raise ValueError('check_interval must be positive')
    if score_batches is not None and not callable(score_batches):
        raise ValueError('score_batches must be callable')

    n_features = model.get_n_features()
    n_classes = model.get_n_classes()

    logits_buf = _make_buffer(n_classes)
    probs_buf = _make_buffer(n_classes)
    bias_buf = _make_buffer(n_classes)

    prev_loss = None
    final_loss = float('inf')
    iterations_completed = 0

    for _ in range(max_iterations):
        iterations_completed += 1
        batches = batch_iter_factory()
        try:
            batch_iter = iter(batches)
        except TypeError:
            raise ValueError('batch iterator must be iterable')

        batches_processed = 0

        for batch in batch_iter:
            batches_processed += 1
            try:
                X_batch, y_batch = batch
            except Exception as exc:
                raise ValueError('each batch must unpack into (X_batch, y_batch)') from exc

            if len(X_batch) == 0:
                continue
            if len(X_batch) % n_features != 0:
                raise ValueError('X_batch size mismatch with n_features')
            n_samples = len(X_batch) // n_features
            if len(y_batch) != n_samples * n_classes:
                raise ValueError('y_batch must be one-hot encoded (len = n_samples * n_classes)')

            model.step(X_batch, y_batch, logits_buf, probs_buf, bias_buf)

        if batches_processed == 0:
            raise ValueError('batch iterator produced no batches')

        if iterations_completed % check_interval != 0:
            continue
        if score_batches is None:
            continue

        current_loss = float(score_batches(model))
        final_loss = current_loss
        change = float('inf') if prev_loss is None else abs(prev_loss - current_loss)

        if verbose >= 2:
            print(log_prefix, f'Iteration {iterations_completed} loss={current_loss}')

        converged = change < tolerance and iterations_completed > check_interval * 2

        if score_limit is not None and current_loss <= score_limit:
            converged = True

        diverged = not (current_loss == current_loss)
        if not diverged and prev_loss is not None:
            diverged = current_loss > prev_loss * divergence_factor

        if converged:
            if verbose >= 1:
                print(log_prefix, f"Converged at iteration {iterations_completed}")
            break

        if diverged:
            if verbose >= 1:
                print(log_prefix, f"Diverged at iteration {iterations_completed}")
            break

        prev_loss = current_loss

    if score_batches is not None and final_loss == float('inf'):
        final_loss = float(score_batches(model))

    return iterations_completed, final_loss

