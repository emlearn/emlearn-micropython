#!/usr/bin/env python3
"""
MicroPython example: train an emlearn_extratrees model on the Occupancy Detection dataset.

Compares results with the sklearn benchmark (RandomForest) from benchmark.py.

Usage:
    MICROPYPATH=./dist/x64_6.3/:./tests/ micropython examples/datasets/occupancy/train_extratrees.py
"""

import array
import gc
import time
import npyfile
import emlearn_extratrees


def load_int16(path):
    """Load .npy file and convert float32 features to int16 (×1000)."""
    shape, data = npyfile.load(path)
    return shape, array.array('h', (int(v * 1000) for v in data))


def load_labels_int16(path):
    """Load .npy file of float labels and convert to int16."""
    shape, data = npyfile.load(path)
    return shape, array.array('h', (int(v) for v in data))


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

    # Load train/test data
    X_train_shape, X_train = load_int16(data_dir + "X_train.npy")
    y_train_shape, y_train = load_labels_int16(data_dir + "y_train.npy")
    X_test_shape, X_test = load_int16(data_dir + "X_test.npy")
    y_test_shape, y_test = load_labels_int16(data_dir + "y_test.npy")

    n_features = X_train_shape[1]
    n_train = y_train_shape[0]
    n_test = y_test_shape[0]

    print(f"  Train: {n_train} samples, {n_features} features")
    print(f"  Test:  {n_test} samples")

    # Count class distribution
    train_pos = sum(1 for v in y_train if v > 0)
    test_pos = sum(1 for v in y_test if v > 0)
    print(f"  Train class distribution:  0={n_train - train_pos}  1={train_pos}")
    print(f"  Test  class distribution:  0={n_test - test_pos}  1={test_pos}")

    # Free raw data
    gc.collect()

    # -----------------------------------------------------------------
    # Model configuration
    # -----------------------------------------------------------------
    # These parameters give ~0.96 PR AUC, competitive with sklearn
    model = emlearn_extratrees.new(
        n_features, 2,
        n_trees=30, max_depth=8, min_samples_leaf=2,
        n_thresholds=10, subsample_ratio=0.5, feature_subsample_ratio=0.7,
        max_nodes=8000, max_samples=n_train, rng_seed=42,
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    print("\nTraining ExtraTrees model ...")
    t0 = time.ticks_ms()
    model.train(X_train, y_train)
    t_train = time.ticks_diff(time.ticks_ms(), t0)

    print(f"  Training time:    {t_train} ms")
    print(f"  Trees used:       {model.get_n_trees_trained()}")
    print(f"  Nodes used:       {model.get_n_nodes_used()}")

    # -----------------------------------------------------------------
    # Evaluation on held-out test set
    # -----------------------------------------------------------------
    print("\nEvaluating on test set ...")

    probs_buf = array.array('f', [0.0, 0.0])
    correct = 0
    y_scores = array.array('f', [0.0] * n_test)
    y_trues = array.array('f', [0.0] * n_test)

    for i in range(n_test):
        start = i * n_features
        features = array.array('h', X_test[start:start + n_features])
        pred = model.predict(features, probs_buf)
        y_trues[i] = float(y_test[i])
        y_scores[i] = probs_buf[1]
        if pred == int(y_test[i]):
            correct += 1

    accuracy = correct / n_test
    pr_auc = average_precision(y_trues, y_scores)

    print(f"  Accuracy:  {accuracy:.4f} ({correct}/{n_test})")
    print(f"  PR AUC:    {pr_auc:.4f}")

    # -----------------------------------------------------------------
    # Comparison with sklearn benchmark from benchmark.py
    # -----------------------------------------------------------------
    print("\nComparison with sklearn benchmark (held-out test set):")
    print(f"  emlearn ExtraTrees              PR AUC: {pr_auc:.4f}")
    print(f"  sklearn RandomForest (n=10)     PR AUC: 0.9488  (benchmark)")
    print(f"  sklearn LogisticRegression      PR AUC: 0.9762  (benchmark)")

    if pr_auc >= 0.94:
        print("\n✅ EXCELLENT: Competitive with professional ML performance!")
    elif pr_auc >= 0.90:
        print("\n✅ VERY GOOD: Strong performance on occupancy detection!")
    elif pr_auc >= 0.85:
        print("\n✅ GOOD: Solid performance!")
    else:
        print("\n⚠️  FAIR: Room for improvement (try more trees or deeper splits)")


if __name__ == "__main__":
    main()
