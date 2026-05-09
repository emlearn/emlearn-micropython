"""Time-series anomaly detection using Linear Regression on NASA SMAP telemetry data.

Demonstrates unsupervised anomaly detection via linear regression with
online learning (EM Learn). The model predicts the target sensor reading from
other correlated sensors and command encodings in the training data. At test
time, large prediction errors indicate anomalous timesteps.

Dataset: NASA SMAP (Soil Moisture Active Passive) telemetry channel P-1
Source: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Paper: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
       Dynamic Thresholding", KDD 2018.

Run with MicroPython (~4 MB heap sufficient):
    micropython -X heapsize=4M run.py
"""

import sys
import array
import gc
import time

# Add directories to sys.path so we can import both modules
sys.path.insert(0, '/workspace/tests')            # npyfile for loading .npy data
sys.path.insert(0, '/workspace/dist/x64_6.3')     # compiled emlearn modules

import emlearn_linreg as lr


# ── Configuration ─────────────────────────────────────────────────────────────

CHANNEL = 'P-1'  # SMAP power channel with 3 known anomaly regions
DATA_DIR = '/workspace/examples/datasets/smap_msl/data_raw'
TRAIN_PATH = DATA_DIR + '/train/' + CHANNEL + '.npy'
TEST_PATH = DATA_DIR + '/test/' + CHANNEL + '.npy'

# Anomaly labels for P-1 (start, end) in test data.
# Source: https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
ANOMALY_LABELS = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844),
]

# Model hyperparameters
LEARNING_RATE = 0.01
L2_REGULARIZATION = 0.0
TRAIN_ITERATIONS = 3000
CHECK_INTERVAL = 10
MSE_TOLERANCE = 0.000001

# Threshold: mean + N * std of training residuals
THRESHOLD_SIGMAS = 4.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_npy(filename):
    """Load .npy file using npyfile, returns (shape, array.array)."""
    import npyfile
    shape, data = npyfile.load(filename)
    return shape, data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=== Time-Series Anomaly Detection (Linear Regression) ===')
    print('Dataset: NASA SMAP channel ' + CHANNEL)
    print('Reference anomaly regions in test data:')
    for start, end in ANOMALY_LABELS:
        print('  [%d, %d]' % (start, end))
    print()

    # --- Load .npy data --------------------------------------------------------
    train_shape, X_train = load_npy(TRAIN_PATH)
    test_shape, X_test = load_npy(TEST_PATH)

    n_samples_train = train_shape[0]
    n_features = train_shape[1]
    n_samples_test = test_shape[0]

    print('Training: %d samples x %d features' % (n_samples_train, n_features))
    print('Test:     %d samples x %d features' % (n_samples_test, n_features))
    gc.collect()

    # The data layout per sample: [target_feature, feature_1, ..., feature_N]
    # We predict target_feature (column 0) from all other features (columns 1..N-1).
    # This models the multivariate sensor correlations learned during normal operation.
    # The .npy files use float64 typecode; convert to float32 for the C module.

    n_input_features = n_features - 1  # features used as input (exclude target)
    print('Model: predict column[0] from columns[1..%d]' % (n_features - 1))

    # --- Convert training data from float64 to float32 -------------------------
    print('Converting float64 -> float32 ...')
    X_train_float = array.array('f')
    y_train_float = array.array('f')

    for i in range(n_samples_train):
        base = i * n_features
        target = float(X_train[base])         # column 0 = target to predict
        inputs = [float(X_train[base + c]) for c in range(1, n_features)]
        y_train_float.append(target)
        X_train_float.extend(inputs)

    print('Training data: %d input features x %d samples' % (n_input_features, n_samples_train))

    # --- Train linear regression -------------------------------------------------
    print()
    print('Training model ...')
    t0 = time.ticks_ms()

    model = lr.new(n_input_features, LEARNING_RATE, L2_REGULARIZATION, 0.1)

    stop_iter, stop_mse = lr.train(
        model,
        X_train_float, y_train_float,
        max_iterations=TRAIN_ITERATIONS,
        check_interval=CHECK_INTERVAL,
        verbose=2,
        tolerance=MSE_TOLERANCE,
    )

    train_ms = time.ticks_diff(time.ticks_ms(), t0)
    print('  Finished %d iterations (MSE=%.6f)' % (stop_iter, stop_mse))
    print('  Training time: %d ms' % train_ms)

    gc.collect()

    # --- Evaluate training residuals ---------------------------------------------
    print()
    print('Computing training residual errors ...')

    sample = array.array('f', (0.0 for _ in range(n_input_features)))
    train_errors = array.array('d')

    for i in range(n_samples_train):
        base = i * n_input_features
        for c in range(n_input_features):
            sample[c] = float(X_train_float[base + c])
        pred = model.predict(sample)
        actual = float(y_train_float[i])
        train_errors.append(abs(pred - actual))

    train_err_mean = sum(train_errors) / len(train_errors)
    train_err_var = sum((e - train_err_mean) ** 2 for e in train_errors) / len(train_errors)
    train_err_std = train_err_var ** 0.5

    print('  Residual error: mean=%.4f, std=%.4f' % (train_err_mean, train_err_std))

    # --- Set detection threshold -------------------------------------------------
    threshold = train_err_mean + THRESHOLD_SIGMAS * train_err_std
    print('  Threshold (mean + %.0f*std): %.4f' % (THRESHOLD_SIGMAS, threshold))
    gc.collect()

    # --- Detect anomalies on test data -------------------------------------------
    print()
    print('Detecting anomalies on test data ...')

    n_predicted_anomalies = 0
    tp = 0   # true positives
    fp = 0   # false positives
    fn = 0   # false negatives

    X_sample = array.array('f', (0.0 for _ in range(n_input_features)))

    for i in range(n_samples_test):
        base = i * n_features
        actual = float(X_test[base])          # column 0 is target

        # Build input features (columns 1..N-1)
        for c in range(n_input_features):
            X_sample[c] = float(X_test[base + c + 1])

        pred = model.predict(X_sample)
        error = abs(pred - actual)
        is_predicted_anomaly = error > threshold

        # Check ground truth label
        is_true_anomaly = False
        for a_start, a_end in ANOMALY_LABELS:
            if a_start <= i <= a_end:
                is_true_anomaly = True
                break

        if is_predicted_anomaly and is_true_anomaly:
            tp += 1
        elif is_predicted_anomaly and not is_true_anomaly:
            fp += 1
        elif not is_predicted_anomaly and is_true_anomaly:
            fn += 1

        if is_predicted_anomaly:
            n_predicted_anomalies += 1

    print('Predicted anomalies: %d / %d' % (n_predicted_anomalies, n_samples_test))

    true_anomalies = sum(end - start + 1 for start, end in ANOMALY_LABELS)
    print('Ground truth anomalies: %d / %d' % (true_anomalies, n_samples_test))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    print()
    print('Precision: %.4f (%d/%d)' % (precision, tp, tp + fp))
    print('Recall:    %.4f (%d/%d)' % (recall, tp, tp + fn))
    print('F1-score:  %.4f' % f1)

    # --- Per-region detection summary -------------------------------------------
    print()
    print('Per-region detection:')
    for start, end in ANOMALY_LABELS:
        region_size = end - start + 1
        region_tp = 0
        for i in range(start, end + 1):
            base = i * n_features
            for c in range(n_input_features):
                X_sample[c] = float(X_test[base + c + 1])
            pred = model.predict(X_sample)
            actual = float(X_test[base])
            if abs(pred - actual) > threshold:
                region_tp += 1
        print('  [%4d, %4d] (%4d timesteps): detected=%d' % (start, end, region_size, region_tp))

    gc.collect()


if __name__ == '__main__':
    main()
