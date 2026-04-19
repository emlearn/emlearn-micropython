import array
import emlearn_logreg


def make_dataset():
    # simple AND gate dataset
    X = array.array('f', [
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    ])
    y = array.array('f', [0, 0, 0, 1])
    return X, y


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
    return X, y


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
    return X, y


def make_high_dimensional_dataset(n_features=12, n_samples=8):
    # alternating labels with sparse informative features
    X = array.array('f')
    y = array.array('f')
    for i in range(n_samples):
        label = 1.0 if i % 2 == 0 else 0.0
        for f in range(n_features):
            if f == (i % n_features):
                value = 2.0 if label == 1.0 else -2.0
            else:
                value = 0.1 * ((f - i) % 3)
            X.append(value)
        y.append(label)
    return X, y


def make_zero_feature_dataset(n_features):
    X = array.array('f', [0.0] * n_features)
    y = array.array('f', [0.0])
    return X, y


def read_weights(model):
    n_features = model.get_n_features()
    buf = array.array('f', [0.0] * n_features)
    model.get_weights(buf)
    return buf


def read_bias(model):
    return model.get_bias()


def assert_raises_value_error(func, message='Expected ValueError'):
    try:
        func()
    except ValueError:
        return
    raise AssertionError(message)


def test_logreg_train_and_predict():
    X, y = make_dataset()
    model = emlearn_logreg.new(2, 0.5, 0.0, 0.0)

    emlearn_logreg.train(model, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=5, batch_size=2)

    pred = model.predict(array.array('f', [1, 1]))
    assert pred > 0.8, pred

    zero_pred = model.predict(array.array('f', [0, 0]))
    assert zero_pred < 0.2, zero_pred


def test_logreg_weight_io_and_probabilities():
    model = emlearn_logreg.new(2, 0.1, 0.0, 0.0)

    manual_weights = array.array('f', [2.5, -1.5])
    model.set_weights(manual_weights)
    model.set_bias(-0.5)

    stored_weights = array.array('f', [0.0, 0.0])
    model.get_weights(stored_weights)
    for expected, actual in zip(manual_weights, stored_weights):
        assert abs(expected - actual) < 1e-6, (expected, actual)

    bias = model.get_bias()
    assert abs(bias + 0.5) < 1e-6, bias

    assert model.predict(array.array('f', [2.0, 0.0])) > 0.5
    assert model.predict(array.array('f', [0.0, 1.0])) < 0.5


def test_logreg_train_minibatch_reduces_loss():
    X, y = make_linearly_separable_dataset()
    model = emlearn_logreg.new(2, 0.4, 0.01, 0.0)

    initial_loss = model.score_logloss(X, y)

    emlearn_logreg.train(model, X, y, max_iterations=600, tolerance=1e-6,
                         check_interval=10, batch_size=1)

    final_loss = model.score_logloss(X, y)
    assert final_loss < initial_loss * 0.7, (initial_loss, final_loss)


def test_logreg_l2_penalty_shrinks_weights():
    X, y = make_linearly_separable_dataset()

    model_no_penalty = emlearn_logreg.new(2, 0.3, 0.0, 0.0)
    model_l2 = emlearn_logreg.new(2, 0.3, 0.5, 0.0)

    emlearn_logreg.train(model_no_penalty, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=10)
    emlearn_logreg.train(model_l2, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=10)

    weights_no_penalty = read_weights(model_no_penalty)
    weights_l2 = read_weights(model_l2)

    norm_no_penalty = sum(w * w for w in weights_no_penalty)
    norm_l2 = sum(w * w for w in weights_l2)

    assert norm_l2 < norm_no_penalty * 0.7, (weights_no_penalty, weights_l2)


def test_logreg_l1_penalty_promotes_sparsity():
    # Start with a single-sample dataset so gradients are deterministic
    X = array.array('f', [1.0, 2.0])
    y = array.array('f', [1.0])

    model_no_l1 = emlearn_logreg.new(2, 0.5, 0.0, 0.0)
    model_l1 = emlearn_logreg.new(2, 0.5, 0.0, 0.5)

    # Set initial weights so updates differ only due to L1 term
    init_weights = array.array('f', [0.2, 0.2])
    model_no_l1.set_weights(init_weights)
    model_l1.set_weights(init_weights)

    emlearn_logreg.train(model_no_l1, X, y, max_iterations=1, check_interval=1)
    emlearn_logreg.train(model_l1, X, y, max_iterations=1, check_interval=1)

    weights_no_l1 = read_weights(model_no_l1)
    weights_l1 = read_weights(model_l1)

    # L1 should push weights closer to zero, even to exact zero for large enough lambda
    assert abs(weights_l1[0]) < abs(weights_no_l1[0])
    assert abs(weights_l1[1]) < abs(weights_no_l1[1])


def test_logreg_handles_ill_conditioned_features():
    X, y = make_ill_conditioned_dataset()

    model = emlearn_logreg.new(2, 0.01, 0.2, 0.0)

    initial_loss = model.score_logloss(X, y)

    emlearn_logreg.train(model, X, y, max_iterations=2000, tolerance=1e-6,
                         check_interval=100, batch_size=4)

    final_loss = model.score_logloss(X, y)
    assert final_loss < initial_loss, (initial_loss, final_loss)
    assert final_loss == final_loss  # not NaN


def test_logreg_high_dimensional_sparse_case():
    X, y = make_high_dimensional_dataset()
    n_features = len(X) // len(y)

    model = emlearn_logreg.new(n_features, 0.1, 0.05, 0.02)

    initial_loss = model.score_logloss(X, y)

    emlearn_logreg.train(model, X, y, max_iterations=600, tolerance=1e-6,
                         check_interval=30, batch_size=4)

    final_loss = model.score_logloss(X, y)

    assert final_loss < initial_loss * 0.8, (initial_loss, final_loss)


def test_logreg_train_validates_dimensions():
    model = emlearn_logreg.new(2, 0.2, 0.0, 0.0)
    X = array.array('f', [0.0, 1.0, 2.0])  # not divisible by n_features
    y = array.array('f', [0.0, 1.0])

    assert_raises_value_error(lambda: emlearn_logreg.train(model, X, y))


def test_logreg_train_requires_targets():
    model = emlearn_logreg.new(1, 0.2, 0.0, 0.0)
    X = array.array('f')
    y = array.array('f')

    assert_raises_value_error(lambda: emlearn_logreg.train(model, X, y))


def test_logreg_warm_start_sets_new_weights_and_bias():
    X, y = make_dataset()
    model = emlearn_logreg.new(2, 0.3, 0.0, 0.0)

    emlearn_logreg.train(model, X, y, max_iterations=20, check_interval=5)
    trained_weights = read_weights(model)
    trained_bias = read_bias(model)

    manual_model = emlearn_logreg.new(2, 0.3, 0.0, 0.0)
    manual_model.set_weights(trained_weights)
    manual_model.set_bias(trained_bias)

    sample = array.array('f', [1.0, 1.0])
    pred_trained = model.predict(sample)
    pred_manual = manual_model.predict(sample)
    assert abs(pred_trained - pred_manual) < 1e-6


def test_logreg_threshold_adjustment_behaviour():
    model = emlearn_logreg.new(2, 0.1, 0.0, 0.0)
    weights = array.array('f', [5.0, -5.0])
    model.set_weights(weights)
    model.set_bias(-1.0)

    features = array.array('f', [0.2, 0.1])
    proba = model.predict(features)
    assert 0.0 < proba < 1.0

    default_label = 1 if proba >= 0.5 else 0
    custom_threshold = 0.3
    custom_label = 1 if proba >= custom_threshold else 0

    assert custom_label >= default_label


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
