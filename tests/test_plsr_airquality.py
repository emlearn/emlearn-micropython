#!/usr/bin/env python3
"""MicroPython test for PLSR on the Air Quality UCI dataset."""

import array
import emlearn_plsr
import npyfile


DATA_DIR = 'examples/datasets/airquality/'


def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y_true, y_pred)) / n


def r2_score(y_true, y_pred):
    n = len(y_true)
    y_mean = sum(y_true) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_true)
    ss_res = sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def test_plsr_airquality():
    """Test PLSR on Air Quality UCI dataset (regression with 13 features)."""
    print("\n=== Air Quality PLSR Test ===")

    # Load data
    shape_X_train, X_train = npyfile.load(DATA_DIR + 'X_train.npy')
    shape_y_train, y_train = npyfile.load(DATA_DIR + 'y_train.npy')
    shape_X_test, X_test = npyfile.load(DATA_DIR + 'X_test.npy')
    shape_y_test, y_test = npyfile.load(DATA_DIR + 'y_test.npy')

    n_train = shape_X_train[0]
    n_features = shape_X_train[1]
    n_test = shape_X_test[0]

    print(f"Loaded: {n_train} train, {n_test} test samples")
    print(f"Features: {n_features}")

    n_components = 3

    # Create and train model
    model = emlearn_plsr.new(n_train, n_features, n_components)
    total_iter, final_metric = emlearn_plsr.fit(
        model, X_train, y_train,
        max_iterations=2000,
        tolerance=1e-5,
        verbose=0,
    )

    assert total_iter > 0, "Some iterations performed"
    assert model.is_complete(), "Training complete"
    print(f"Trained: {total_iter} iterations")

    # Predict on test set
    y_pred = array.array('f')
    for i in range(n_test):
        row = X_test[i * n_features:(i + 1) * n_features]
        y_pred.append(model.predict(row))

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.5f}")
    print(f"Test R^2: {r2:.5f}")
    print(f"Target (sklearn PLSR): ~0.97")

    # emlearn PLSR should be close to sklearn (which gets ~0.977)
    assert r2 > 0.90, "R^2 above 0.90"

    if r2 >= 0.90:
        print("✅ GOOD: Solid regression performance on real data!")
    else:
        print("❌ POOR: R^2 below threshold")


if __name__ == '__main__':
    test_plsr_airquality()
