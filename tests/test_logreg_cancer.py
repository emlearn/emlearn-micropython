import array
import gc
import emlearn_logreg
import npyfile


def load_flattened(path):
    shape, buf = npyfile.load(path)
    return shape, array.array('f', buf)


def one_hot(labels, n_classes):
    out = array.array('f', [0.0] * (len(labels) * n_classes))
    for idx, label in enumerate(labels):
        out[idx * n_classes + int(label)] = 1.0
    return out


def argmax_array(buf):
    best_idx = 0
    best_val = buf[0]
    for idx in range(1, len(buf)):
        if buf[idx] > best_val:
            best_val = buf[idx]
            best_idx = idx
    return best_idx


def accuracy_on_dataset(model, X, y, n_features):
    correct = 0
    n_classes = model.get_n_classes()
    n_samples = len(X) // n_features
    logits = array.array('f', [0.0] * n_classes)
    probs = array.array('f', [0.0] * n_classes)
    for idx in range(n_samples):
        start = idx * n_features
        features = array.array('f', X[start:start + n_features])
        model.predict(features, probs, logits)
        pred = argmax_array(probs)
        target = y[idx * n_classes:(idx + 1) * n_classes]
        true_label = argmax_array(target)
        if pred == true_label:
            correct += 1
    return correct / n_samples


def test_logreg_real_dataset_binary_classification():

    data_dir = 'examples/datasets/cancer/'
    DATA_FILES = {
        'X_train': data_dir+'X_train.npy',
        'X_test': data_dir+'X_test.npy',
        'y_train': data_dir+'y_train.npy',
        'y_test': data_dir+'y_test.npy',
    }

    gc.collect()
    X_train_shape, X_train = load_flattened(DATA_FILES['X_train'])
    y_train_shape, y_train = load_flattened(DATA_FILES['y_train'])
    X_test_shape, X_test = load_flattened(DATA_FILES['X_test'])
    y_test_shape, y_test = load_flattened(DATA_FILES['y_test'])

    n_features = X_train_shape[1]
    n_train = y_train_shape[0]
    n_test = y_test_shape[0]

    assert len(X_train) == n_train * n_features
    assert len(X_test) == n_test * n_features

    n_classes = int(max(y_train)) + 1
    model = emlearn_logreg.new(n_features, n_classes, 0.05, 0.001, 0.0005)

    stop_iter, stop_loss = emlearn_logreg.train(
        model,
        X_train,
        one_hot(y_train, n_classes),
        max_iterations=1500,
        tolerance=1e-5,
        check_interval=25,
        batch_size=64,
        score_limit=0.28,
    )

    assert stop_iter > 0
    assert stop_loss == stop_loss  # not NaN

    logits = array.array('f', [0.0] * n_classes)
    probs = array.array('f', [0.0] * n_classes)

    y_train_oh = one_hot(y_train, n_classes)
    y_test_oh = one_hot(y_test, n_classes)

    train_loss = model.score_logloss(X_train, y_train_oh, logits, probs)
    test_loss = model.score_logloss(X_test, y_test_oh, logits, probs)

    assert train_loss < 0.35, train_loss
    assert test_loss < 0.4, test_loss

    accuracy = accuracy_on_dataset(model, X_test, y_test_oh, n_features)
    assert accuracy > 0.8, accuracy


if __name__ == '__main__':
    test_logreg_real_dataset_binary_classification()
