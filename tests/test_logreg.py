import array
import emlearn_logreg


def to_one_hot(labels, n_classes):
    out = array.array('f', [0.0] * (len(labels) * n_classes))
    for idx, label in enumerate(labels):
        class_idx = int(label)
        out[idx * n_classes + class_idx] = 1.0
    return out


def make_dataset():
    # simple AND gate dataset
    X = array.array('f', [
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    ])
    y = array.array('f', [0, 0, 0, 1])
    return X, to_one_hot(y, 2)


def make_linearly_separable_dataset():
    # two well-separated clusters in 2D
    points = [
        (-2.0, -1.0, 0.0),
        (-1.5, -2.0, 0.0),
        (-1.0, -1.5, 0.0),
        (1.0, 1.5, 1.0),
        (1.5, 1.0, 1.0),
        (2.0, 1.5, 1.0),
    ]
    X = array.array('f')
    y = array.array('f')
    for f1, f2, label in points:
        X.append(f1)
        X.append(f2)
        y.append(label)
    return X, to_one_hot(y, 2)


def make_ill_conditioned_dataset():
    # feature 0 is tiny; feature 1 has large magnitude swings, but labels depend on the combination
    points = [
        (0.0005, 800.0, 1.0),
        (0.0007, 600.0, 1.0),
        (0.0009, 400.0, 1.0),
        (0.0015, -400.0, 0.0),
        (0.0018, -600.0, 0.0),
        (0.0021, -800.0, 0.0),
        (0.0006, 200.0, 1.0),
        (0.0025, -200.0, 0.0),
    ]
    X = array.array('f')
    y = array.array('f')
    for f1, f2, label in points:
        X.append(f1)
        X.append(f2)
        y.append(label)
    return X, to_one_hot(y, 2)


def make_high_dimensional_dataset(n_features=12, n_samples=8):
    # alternating labels with sparse informative features
    X = array.array('f')
    y = array.array('f')
    for i in range(n_samples):
        label = 1 if i % 2 == 0 else 0
        for f in range(n_features):
            if f == (i % n_features):
                value = 2.0 if label == 1 else -2.0
            else:
                value = 0.1 * ((f - i) % 3)
            X.append(value)
        y.append(label)
    return X, to_one_hot(y, 2)


def make_zero_feature_dataset(n_features):
    X = array.array('f', [0.0] * n_features)
    y = array.array('f', [0.0])
    return X, to_one_hot(y, 2)


def make_multiclass_dataset():
    # three clusters in 2D
    points = [
        (-2.0, -2.0, 0.0),
        (-1.5, -1.7, 0.0),
        (-2.2, -1.6, 0.0),
        (2.2, -2.0, 1.0),
        (1.8, -1.4, 1.0),
        (2.5, -1.7, 1.0),
        (0.0, 2.5, 2.0),
        (0.4, 2.0, 2.0),
        (-0.5, 2.1, 2.0),
    ]
    X = array.array('f')
    y = array.array('f')
    for f1, f2, label in points:
        X.append(f1)
        X.append(f2)
        y.append(label)
    return X, to_one_hot(y, 3)


def read_weights(model):
    n_features = model.get_n_features()
    n_classes = model.get_n_classes()
    buf = array.array('f', [0.0] * (n_features * n_classes))
    model.get_weights(buf)
    return buf


def read_bias(model):
    n_classes = model.get_n_classes()
    src = model.get_bias()
    out = array.array('f', [0.0] * n_classes)
    for idx in range(n_classes):
        out[idx] = src[idx]
    return out


def alloc_predict_buffers(model):
    n_classes = model.get_n_classes()
    logits = array.array('f', [0.0] * n_classes)
    probs = array.array('f', [0.0] * n_classes)
    return logits, probs


def argmax_array(buf):
    best_idx = 0
    best_val = buf[0]
    for idx in range(1, len(buf)):
        if buf[idx] > best_val:
            best_val = buf[idx]
            best_idx = idx
    return best_idx


def run_predict(model, features, logits, probs):
    model.predict(features, probs, logits)
    return probs


def predict_binary_class(model, features, threshold=0.5):
    logits, probs = alloc_predict_buffers(model)
    run_predict(model, features, logits, probs)
    proba = probs[1]
    return 1 if proba >= threshold else 0


def assert_raises_value_error(func, message='Expected ValueError'):
    try:
        func()
    except ValueError:
        return
    raise AssertionError(message)


def test_logreg_train_and_predict():
    X, y = make_dataset()
    model = emlearn_logreg.new(2, 2, 0.5, 0.0, 0.0)

    emlearn_logreg.train(model, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=5, batch_size=2)

    logits, probs = alloc_predict_buffers(model)
    run_predict(model, array.array('f', [1, 1]), logits, probs)
    pred = argmax_array(probs)
    assert pred == 1, pred

    run_predict(model, array.array('f', [0, 0]), logits, probs)
    zero_pred = argmax_array(probs)
    assert zero_pred == 0, zero_pred


def test_logreg_weight_io_and_probabilities():
    model = emlearn_logreg.new(2, 2, 0.1, 0.0, 0.0)

    manual_weights = array.array('f', [2.5, -1.5, 1.0, -0.5])
    model.set_weights(manual_weights)
    manual_bias = array.array('f', [-0.5, 0.25])
    model.set_bias(manual_bias)

    stored_weights = array.array('f', [0.0] * len(manual_weights))
    model.get_weights(stored_weights)
    for expected, actual in zip(manual_weights, stored_weights):
        assert abs(expected - actual) < 1e-6, (expected, actual)

    stored_bias = read_bias(model)
    for expected, actual in zip(manual_bias, stored_bias):
        assert abs(expected - actual) < 1e-6, (expected, actual)

    logits, probs = alloc_predict_buffers(model)
    run_predict(model, array.array('f', [2.0, 0.0]), logits, probs)
    run_predict(model, array.array('f', [0.0, 1.0]), logits, probs)


def test_logreg_train_minibatch_reduces_loss():
    X, y = make_linearly_separable_dataset()
    model = emlearn_logreg.new(2, 2, 0.4, 0.01, 0.0)

    logits, probs = alloc_predict_buffers(model)
    initial_loss = model.score_logloss(X, y, logits, probs)

    emlearn_logreg.train(model, X, y, max_iterations=600, tolerance=1e-6,
                         check_interval=10, batch_size=1)

    final_loss = model.score_logloss(X, y, logits, probs)
    assert final_loss < initial_loss * 0.7, (initial_loss, final_loss)


def test_logreg_l2_penalty_shrinks_weights():
    X, y = make_linearly_separable_dataset()

    model_no_penalty = emlearn_logreg.new(2, 2, 0.3, 0.0, 0.0)
    model_l2 = emlearn_logreg.new(2, 2, 0.3, 0.5, 0.0)

    emlearn_logreg.train(model_no_penalty, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=10)
    emlearn_logreg.train(model_l2, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=10)

    logits, probs = alloc_predict_buffers(model_no_penalty)
    initial_loss = model_no_penalty.score_logloss(X, y, logits, probs)
    final_loss = model_l2.score_logloss(X, y, logits, probs)

    weights_no_penalty = read_weights(model_no_penalty)
    weights_l2 = read_weights(model_l2)

    norm_no_penalty = sum(w * w for w in weights_no_penalty)
    norm_l2 = sum(w * w for w in weights_l2)

    assert norm_l2 < norm_no_penalty * 0.7, (weights_no_penalty, weights_l2)


def test_logreg_l1_penalty_promotes_sparsity():
    # Start with a single-sample dataset so gradients are deterministic
    X = array.array('f', [1.0, 2.0])
    y = to_one_hot(array.array('f', [1.0]), 2)

    model_no_l1 = emlearn_logreg.new(2, 2, 0.5, 0.0, 0.0)
    model_l1 = emlearn_logreg.new(2, 2, 0.5, 0.0, 0.5)

    # Set initial weights so updates differ only due to L1 term
    init_weights = array.array('f', [0.2, 0.2, 0.2, 0.2])
    model_no_l1.set_weights(init_weights)
    model_l1.set_weights(init_weights)

    emlearn_logreg.train(model_no_l1, X, y, max_iterations=1, check_interval=1)
    emlearn_logreg.train(model_l1, X, y, max_iterations=1, check_interval=1)

    weights_no_l1 = read_weights(model_no_l1)
    weights_l1 = read_weights(model_l1)

    # Compare each feature weight for both classes
    for idx in range(len(weights_no_l1)):
        assert abs(weights_l1[idx]) < abs(weights_no_l1[idx])


def test_logreg_handles_ill_conditioned_features():
    X, y = make_ill_conditioned_dataset()

    model = emlearn_logreg.new(2, 2, 0.01, 0.2, 0.0)

    logits, probs = alloc_predict_buffers(model)
    initial_loss = model.score_logloss(X, y, logits, probs)

    emlearn_logreg.train(model, X, y, max_iterations=2000, tolerance=1e-6,
                         check_interval=100, batch_size=4)

    final_loss = model.score_logloss(X, y, logits, probs)
    assert final_loss < initial_loss, (initial_loss, final_loss)
    assert final_loss == final_loss  # not NaN


def test_logreg_high_dimensional_sparse_case():
    X, y = make_high_dimensional_dataset()
    n_classes = 2
    n_samples = len(y) // n_classes
    n_features = len(X) // n_samples

    model = emlearn_logreg.new(n_features, n_classes, 0.1, 0.05, 0.02)

    logits, probs = alloc_predict_buffers(model)
    initial_loss = model.score_logloss(X, y, logits, probs)

    emlearn_logreg.train(model, X, y, max_iterations=600, tolerance=1e-6,
                         check_interval=30, batch_size=4)

    final_loss = model.score_logloss(X, y, logits, probs)

    assert final_loss < initial_loss * 0.8, (initial_loss, final_loss)


def test_logreg_train_validates_dimensions():
    model = emlearn_logreg.new(2, 2, 0.2, 0.0, 0.0)
    X = array.array('f', [0.0, 1.0, 2.0])  # not divisible by n_features
    y = array.array('f', [0.0, 1.0])

    assert_raises_value_error(lambda: emlearn_logreg.train(model, X, y))


def test_logreg_train_requires_targets():
    model = emlearn_logreg.new(1, 2, 0.2, 0.0, 0.0)
    X = array.array('f')
    y = array.array('f')

    assert_raises_value_error(lambda: emlearn_logreg.train(model, X, y))


def test_logreg_warm_start_sets_new_weights_and_bias():
    X, y = make_dataset()
    model = emlearn_logreg.new(2, 2, 0.3, 0.0, 0.0)

    emlearn_logreg.train(model, X, y, max_iterations=20, check_interval=5)
    trained_weights = read_weights(model)
    trained_bias = read_bias(model)

    manual_model = emlearn_logreg.new(2, 2, 0.3, 0.0, 0.0)
    manual_model.set_weights(trained_weights)
    manual_model.set_bias(trained_bias)

    sample = array.array('f', [1.0, 1.0])
    logits, probs = alloc_predict_buffers(model)
    run_predict(model, sample, logits, probs)
    trained_probs = probs[:]
    run_predict(manual_model, sample, logits, probs)
    for p1, p2 in zip(trained_probs, probs):
        assert abs(p1 - p2) < 1e-6


def test_logreg_multiclass_softmax_train_set_accuracy():
    X, y = make_multiclass_dataset()
    n_features = 2
    n_classes = 3
    model = emlearn_logreg.new(n_features, n_classes, 0.2, 0.01, 0.0)

    emlearn_logreg.train(
        model,
        X,
        y,
        max_iterations=1200,
        tolerance=1e-6,
        check_interval=60,
        batch_size=3,
    )

    for idx in range(len(y) // n_classes):
        start = idx * n_features
        features = array.array('f', X[start:start + n_features])
        logits, probs = alloc_predict_buffers(model)
        run_predict(model, features, logits, probs)
        pred = argmax_array(probs)
        label_slice = y[idx * n_classes:(idx + 1) * n_classes]
        expected = argmax_array(label_slice)
        assert pred == expected, (pred, expected)


def test_logreg_multiclass_softmax_generalization():
    X, y = make_multiclass_dataset()
    n_features = 2
    n_classes = 3
    model = emlearn_logreg.new(n_features, n_classes, 0.2, 0.01, 0.0)

    emlearn_logreg.train(
        model,
        X,
        y,
        max_iterations=1200,
        tolerance=1e-6,
        check_interval=60,
        batch_size=3,
    )

    test_points = [
        (array.array('f', [-2.1, -1.9]), 0),
        (array.array('f', [2.3, -1.8]), 1),
        (array.array('f', [-0.2, 2.4]), 2),
    ]
    logits, probs = alloc_predict_buffers(model)
    for features, expected in test_points:
        run_predict(model, features, logits, probs)
        pred = argmax_array(probs)
        assert pred == expected, (pred, expected)


if __name__ == '__main__':
    test_logreg_train_and_predict()
    test_logreg_weight_io_and_probabilities()
    test_logreg_train_minibatch_reduces_loss()
    test_logreg_l2_penalty_shrinks_weights()
    test_logreg_l1_penalty_promotes_sparsity()
    test_logreg_handles_ill_conditioned_features()
    test_logreg_high_dimensional_sparse_case()
    test_logreg_train_validates_dimensions()
    test_logreg_train_requires_targets()
    test_logreg_warm_start_sets_new_weights_and_bias()
    test_logreg_one_vs_rest_classifies_training_samples()
    test_logreg_one_vs_rest_generalizes_new_points()
