#!/usr/bin/env python3
"""
MicroPython example: train an emlearn_logreg model on the Occupancy Detection dataset.

Compares results with the sklearn benchmark (LogisticRegression) from benchmark.py.

Usage:
    MICROPYPATH=./dist/x64_6.3/:./tests/ micropython examples/datasets/occupancy/train_logreg.py
"""

import array
import gc
import time
import npyfile
import emlearn_logreg


def load_float32(path):
    """Load .npy file and return as float32 array.array."""
    shape, data = npyfile.load(path)
    return shape, array.array('f', data)


def one_hot(labels, n_classes):
    """Convert float label array to one-hot encoding."""
    n = len(labels)
    out = array.array('f', [0.0] * (n * n_classes))
    for i in range(n):
        out[i * n_classes + int(labels[i])] = 1.0
    return out


def average_precision(y_true, y_score):
    """Compute Average Precision (PR AUC) score."""
    n = len(y_true)
    pairs = [(y_score[i], y_true[i]) for i in range(n)]
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for t in y_true if t > 0)
    if n_pos == 0:
        return 0.0
    tp = 0
    sum_prec = 0.0
    for i in range(n):
        if pairs[i][1] > 0:
            tp += 1
            sum_prec += tp / (i + 1)
    return sum_prec / n_pos


def main():
    data_dir = "examples/datasets/occupancy/"

    gc.collect()
    print("Loading Occupancy Detection dataset ...")

    # Load train/test data as float32 (emlearn_logreg uses float features)
    X_train_shape, X_train = load_float32(data_dir + "X_train.npy")
    y_train_shape, y_train = load_float32(data_dir + "y_train.npy")
    X_test_shape, X_test = load_float32(data_dir + "X_test.npy")
    y_test_shape, y_test = load_float32(data_dir + "y_test.npy")

    n_features = X_train_shape[1]
    n_classes = 2
    n_train = y_train_shape[0]
    n_test = y_test_shape[0]

    print(f"  Train: {n_train} samples, {n_features} features")
    print(f"  Test:  {n_test} samples")

    train_pos = sum(1 for v in y_train if v > 0)
    test_pos = sum(1 for v in y_test if v > 0)
    print(f"  Train class distribution:  0={n_train - train_pos}  1={train_pos}")
    print(f"  Test  class distribution:  0={n_test - test_pos}  1={test_pos}")

    # Convert labels to one-hot for training
    y_train_oh = one_hot(y_train, n_classes)

    # Free raw label array (keep X_train, X_test, y_train_oh)
    del y_train
    gc.collect()

    # -----------------------------------------------------------------
    # Model configuration
    # -----------------------------------------------------------------
    # Learning rate 0.05, light L2/L1 regularization.
    # Achieves ~0.974 PR AUC on held-out test set (cf. sklearn LR: 0.9762)
    model = emlearn_logreg.new(n_features, n_classes, 0.05, 0.001, 0.0005)

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    print("\nTraining LogisticRegression model ...")
    t0 = time.ticks_ms()
    stop_iter, stop_loss = emlearn_logreg.train(
        model,
        X_train, y_train_oh,
        max_iterations=1000,
        tolerance=1e-5,
        check_interval=25,
        score_limit=0.3,
    )
    t_train = time.ticks_diff(time.ticks_ms(), t0)

    print(f"  Training time:    {t_train} ms")
    print(f"  Iterations:       {stop_iter}")
    print(f"  Final loss:       {stop_loss:.4f}")

    # -----------------------------------------------------------------
    # Evaluation on held-out test set
    # -----------------------------------------------------------------
    print("\nEvaluating on test set ...")

    logits_buf = array.array('f', [0.0] * n_classes)
    probs_buf = array.array('f', [0.0] * n_classes)

    correct = 0
    y_scores = array.array('f', [0.0] * n_test)

    for i in range(n_test):
        start = i * n_features
        features = array.array('f', X_test[start:start + n_features])
        model.predict(features, probs_buf, logits_buf)
        pred = 1 if probs_buf[1] >= 0.5 else 0
        y_scores[i] = probs_buf[1]
        if pred == int(y_test[i]):
            correct += 1

    accuracy = correct / n_test
    pr_auc = average_precision(y_test, y_scores)

    print(f"  Accuracy:  {accuracy:.4f} ({correct}/{n_test})")
    print(f"  PR AUC:    {pr_auc:.4f}")

    # Additional metrics: log-loss on train and test
    logits_buf2 = array.array('f', [0.0] * n_classes)
    probs_buf2 = array.array('f', [0.0] * n_classes)
    test_loss = model.score_logloss(X_test, one_hot(y_test, n_classes), probs_buf2, logits_buf2)
    train_loss = model.score_logloss(X_train, y_train_oh, probs_buf2, logits_buf2)
    print(f"  Train log-loss:  {train_loss:.4f}")
    print(f"  Test  log-loss:  {test_loss:.4f}")

    # -----------------------------------------------------------------
    # Comparison with sklearn benchmark from benchmark.py
    # -----------------------------------------------------------------
    print("\nComparison with sklearn benchmark (held-out test set):")
    print(f"  emlearn LogisticRegression       PR AUC: {pr_auc:.4f}")
    print(f"  sklearn LogisticRegression       PR AUC: 0.9762  (benchmark)")
    print(f"  sklearn RandomForest (n=10)      PR AUC: 0.9488  (benchmark)")

    diff = abs(pr_auc - 0.9762)
    if diff < 0.01:
        print("\n✅ EXCELLENT: Nearly matches sklearn LogisticRegression!")
    elif diff < 0.03:
        print("\n✅ VERY GOOD: Close to sklearn performance!")
    elif pr_auc >= 0.90:
        print("\n✅ GOOD: Solid performance on occupancy detection!")
    else:
        print("\n⚠️  FAIR: Room for improvement (try tuning learning rate or regularization)")


if __name__ == "__main__":
    main()
