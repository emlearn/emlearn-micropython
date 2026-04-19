import array
import gc
import emlearn_logreg
import npyfile

DATA_FILES = {
    'X_train': 'examples/datasets/wine/X_train.npy',
    'X_test': 'examples/datasets/wine/X_test.npy',
    'y_train': 'examples/datasets/wine/y_train.npy',
    'y_test': 'examples/datasets/wine/y_test.npy',
}


def one_hot_array(labels, n_classes):
    out = array.array('f', [0.0] * (len(labels) * n_classes))
    for idx, label in enumerate(labels):
        out[idx * n_classes + int(label)] = 1.0
    return out


def iter_npy_batches(X_path, y_path, batch_size, n_features, n_classes):
    def factory():
        with npyfile.Reader(X_path) as X_reader, npyfile.Reader(y_path) as y_reader:
            samples = X_reader.shape[0]
            assert y_reader.shape[0] == samples
            X_chunks = X_reader.read_data_chunks(batch_size * n_features)
            y_chunks = y_reader.read_data_chunks(batch_size)
            for X_chunk, y_chunk in zip(X_chunks, y_chunks):
                count = len(y_chunk)
                if count == 0:
                    continue
                assert len(X_chunk) == count * n_features
                yield X_chunk, one_hot_array(y_chunk, n_classes)
    return factory


def stream_score_batches(model, X_path, y_path, n_features, n_classes):
    logits = array.array('f', [0.0] * n_classes)
    probs = array.array('f', [0.0] * n_classes)

    def scorer(model_instance):
        with npyfile.Reader(X_path) as X_reader, npyfile.Reader(y_path) as y_reader:
            total_loss = 0.0
            total_samples = 0
            X_chunks = X_reader.read_data_chunks(64 * n_features)
            y_chunks = y_reader.read_data_chunks(64)
            for X_chunk, y_chunk in zip(X_chunks, y_chunks):
                count = len(y_chunk)
                if count == 0:
                    continue
                total_samples += count
                y_oh = one_hot_array(y_chunk, n_classes)
                total_loss += model_instance.score_logloss(X_chunk, y_oh, logits, probs) * count
            return total_loss / max(total_samples, 1)

    return scorer


def test_logreg_stream_batches_accuracy():
    gc.collect()
    X_shape, X_train_full = npyfile.load(DATA_FILES['X_train'])
    y_shape, y_train = npyfile.load(DATA_FILES['y_train'])
    n_features = X_shape[1]
    n_classes = int(max(y_train)) + 1
    y_train_oh = one_hot_array(y_train, n_classes)
    assert len(X_train_full) == len(y_train) * n_features

    model = emlearn_logreg.new(n_features, n_classes, 0.05, 0.001, 0.0005)

    def batch_factory():
        return iter_npy_batches(DATA_FILES['X_train'], DATA_FILES['y_train'], 32, n_features, n_classes)()

    def scorer(model_instance):
        logits = array.array('f', [0.0] * n_classes)
        probs = array.array('f', [0.0] * n_classes)
        return model_instance.score_logloss(X_train_full, y_train_oh, logits, probs)

    logits_all = array.array('f', [0.0] * n_classes)
    probs_all = array.array('f', [0.0] * n_classes)
    baseline_loss = model.score_logloss(X_train_full, y_train_oh, logits_all, probs_all)

    emlearn_logreg.train_batches(
        model,
        batch_factory,
        max_iterations=200,
        tolerance=1e-5,
        check_interval=10,
        divergence_factor=2.0,
        score_limit=0.35,
        score_batches=scorer,
    )

    final_loss = model.score_logloss(X_train_full, y_train_oh, logits_all, probs_all)
    assert final_loss < baseline_loss, (baseline_loss, final_loss)

    X_test_shape, X_test = npyfile.load(DATA_FILES['X_test'])
    y_test_shape, y_test = npyfile.load(DATA_FILES['y_test'])
    n_features_test = X_test_shape[1]
    assert n_features_test == n_features
    y_test_oh = one_hot_array(y_test, n_classes)
    logits = array.array('f', [0.0] * n_classes)
    probs = array.array('f', [0.0] * n_classes)
    test_loss = model.score_logloss(X_test, y_test_oh, logits, probs)
    assert test_loss < 0.45, test_loss


if __name__ == '__main__':
    test_logreg_stream_batches_accuracy()
