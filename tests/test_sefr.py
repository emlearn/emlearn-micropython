import array
import emlearn_sefr
import npyfile


def load_flattened(path):
    shape, buf = npyfile.load(path)
    return shape, array.array('f', buf)


def accuracy(model, X, y, n_features):
    scores = array.array('f', [0.0] * model.get_n_classes())
    correct = 0
    for i in range(len(y)):
        features = array.array('f', X[i * n_features:(i + 1) * n_features])
        correct += int(model.predict(features, scores) == int(y[i]))
    return correct / len(y)


def test_sefr_separable():
    # three well separated clusters in 2d
    X = array.array('f', [0, 0, 0, 1, 5, 5, 6, 5, 9, 0, 10, 1])
    y = array.array('f', [0, 0, 1, 1, 2, 2])
    model = emlearn_sefr.new(2, 3)
    emlearn_sefr.train(model, X, y)
    assert accuracy(model, X, y, 2) == 1.0


def test_sefr_partial_fit_matches_single_batch():
    X = array.array('f', [0.5, 2, 1, 3, 4, 1, 5, 0, 2, 2, 6, 1, 1, 4])
    y = array.array('f', [0, 0, 1, 1, 0, 1, 0])
    full = emlearn_sefr.new(2, 2)
    emlearn_sefr.train(full, X, y)

    chunked = emlearn_sefr.new(2, 2)
    for start in range(0, len(y), 3):
        end = min(start + 3, len(y))
        chunked.partial_fit(X[start * 2:end * 2], y[start:end])
    chunked.finalize()

    w_full = array.array('f', [0.0] * 4)
    w_chunked = array.array('f', [0.0] * 4)
    full.get_weights(w_full)
    chunked.get_weights(w_chunked)
    for a, b in zip(w_full, w_chunked):
        assert abs(a - b) < 1e-6, (a, b)


def test_sefr_errors():
    model = emlearn_sefr.new(2, 2)
    scores = array.array('f', [0.0, 0.0])
    try:
        model.predict(array.array('f', [0, 0]), scores)
        assert False, 'predict before finalize should fail'
    except ValueError:
        pass
    try:
        model.partial_fit(array.array('f', [0, 0]), array.array('f', [2]))
        assert False, 'out of range label should fail'
    except ValueError:
        pass
    model.partial_fit(array.array('f', [0, 0, 1, 1]), array.array('f', [0, 0]))
    try:
        model.finalize()
        assert False, 'a single class should fail'
    except ValueError:
        pass


def test_sefr_lda_streamed_matches_single_batch():
    X = array.array('f', [0.5, 2, 1, 3, 4, 1, 5, 0, 2, 2, 6, 1, 1, 4, 3, 3, 0, 5])
    y = array.array('f', [0, 0, 1, 1, 2, 1, 2, 0, 0])
    full = emlearn_sefr.new(2, 3)
    emlearn_sefr.train(full, X, y, lda=True)

    chunked = emlearn_sefr.new(2, 3)
    for second in (False, True):
        fit = chunked.partial_fit_lda if second else chunked.partial_fit
        for start in range(0, len(y), 4):
            end = min(start + 4, len(y))
            fit(X[start * 2:end * 2], y[start:end])
        chunked.finalize_lda() if second else chunked.finalize()

    w_full = array.array('f', [0.0] * 6)
    w_chunked = array.array('f', [0.0] * 6)
    full.get_weights(w_full)
    chunked.get_weights(w_chunked)
    for a, b in zip(w_full, w_chunked):
        assert abs(a - b) < 1e-4 * max(1.0, abs(a)), (a, b)


def check_dataset(name, min_accuracy, lda=False):
    d = 'examples/datasets/' + name + '/'
    (n_train, n_features), X_train = load_flattened(d + 'X_train.npy')
    _, y_train = load_flattened(d + 'y_train.npy')
    _, X_test = load_flattened(d + 'X_test.npy')
    _, y_test = load_flattened(d + 'y_test.npy')
    model = emlearn_sefr.new(n_features, int(max(y_train)) + 1)
    emlearn_sefr.train(model, X_train, y_train, lda=lda)
    acc = accuracy(model, X_test, y_test, n_features)
    assert acc >= min_accuracy, (name, lda, acc)


def test_sefr_cancer():
    check_dataset('cancer', 0.9)


def test_sefr_wine():
    check_dataset('wine', 0.9)


def test_sefr_lda_wine():
    check_dataset('wine', 0.9, lda=True)


if __name__ == '__main__':
    test_sefr_separable()
    test_sefr_partial_fit_matches_single_batch()
    test_sefr_errors()
    test_sefr_cancer()
    test_sefr_wine()
    test_sefr_lda_streamed_matches_single_batch()
    test_sefr_lda_wine()
