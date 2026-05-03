#!/usr/bin/env python3
"""MicroPython test for PLSR on the SpectroFood dataset."""

import array
import emlearn_plsr
import npyfile
import gc


DATA_DIR = 'examples/datasets/spectrofood/'


def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y_true, y_pred)) / n


def r2_score(y_true, y_pred):
    n = len(y_true)
    y_mean = sum(y_true) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_true)
    ss_res = sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def test_plsr_spectrofood():
    """Test PLSR on SpectroFood dataset (regression with 421 spectral features)."""
    print("\n=== SpectroFood PLSR Test ===")

    total_ram = gc.mem_alloc() + gc.mem_free()
    if total_ram < 2000_000:
        print("SKIP: insufficient RAM")
        return

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

    n_components = 5

    # Create and train model
    model = emlearn_plsr.new(n_train, n_features, n_components)
    total_iter, final_metric = emlearn_plsr.fit(
        model, X_train, y_train,
        max_iterations=100,
        tolerance=1e-6,
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
    print(f"Target (sklearn PLSR): ~0.70")

    # emlearn PLSR should be close to sklearn
    assert r2 > 0.60, "R^2 above 0.60"

    if r2 >= 0.65:
        print("✅ GOOD: Regression performance matches sklearn on high-dimensional spectral data!")
    elif r2 >= 0.60:
        print("⚠️  FAIR: Close to sklearn but slightly below")


if __name__ == '__main__':
    test_plsr_spectrofood()
