import array
import gc
import emlearn_logreg
import npyfile


def load_flattened(path):
    shape, buf = npyfile.load(path)
    return shape, array.array('f', buf)


def predict_class_from_proba(model, features, threshold=0.5):
    proba = model.predict(features)
    return 1 if proba >= threshold else 0


def accuracy_on_dataset(model, X, y, n_features, threshold=0.5):
    correct = 0
    n_samples = len(y)
    for idx in range(n_samples):
        start = idx * n_features
        features = array.array('f', X[start:start + n_features])
        pred = predict_class_from_proba(model, features, threshold)
        if pred == int(y[idx]):
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

    model = emlearn_logreg.new(n_features, 0.05, 0.001, 0.0005)

    stop_iter, stop_loss = emlearn_logreg.train(
        model,
        X_train,
        y_train,
        max_iterations=1500,
        tolerance=1e-5,
        check_interval=25,
        batch_size=64,
        score_limit=0.28,
    )

    assert stop_iter > 0
    assert stop_loss == stop_loss  # not NaN

    train_loss = model.score_logloss(X_train, y_train)
    test_loss = model.score_logloss(X_test, y_test)

    assert train_loss < 0.35, train_loss
    assert test_loss < 0.4, test_loss

    accuracy = accuracy_on_dataset(model, X_test, y_test, n_features)
    assert accuracy > 0.9, accuracy


if __name__ == '__main__':
    test_logreg_real_dataset_binary_classification()
