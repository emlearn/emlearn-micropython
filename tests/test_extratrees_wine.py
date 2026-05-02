#!/usr/bin/env python3
# Wine dataset test for MicroPython (sklearn Wine - 3 class classification)

import array
import gc
import time
import npyfile
import emlearn_extratrees

DATA_DIR = 'examples/datasets/wine/'
DATA_FILES = {
    'X_train': DATA_DIR + 'X_train.npy',
    'y_train': DATA_DIR + 'y_train.npy',
    'X_test': DATA_DIR + 'X_test.npy',
    'y_test': DATA_DIR + 'y_test.npy',
}

def load_npy_features_int16(filename):
    """Load .npy file and convert to int16 array (scaled from float32)"""
    shape, data = npyfile.load(filename)
    # Scale float32 data to int16 range (multiply by 1000 and convert)
    scaled = [int(v * 1000) for v in data]
    return array.array('h', scaled)

def load_npy_labels_int16(filename):
    """Load .npy file and convert to int16 array (labels, no scaling)"""
    shape, data = npyfile.load(filename)
    # Labels are already integers (0.0, 1.0, 2.0), just convert directly
    labels = [int(v) for v in data]
    return array.array('h', labels)

def test_wine():
    print("=== WINE DATASET TEST (3-class) ===")
    
    # Load preprocessed data
    try:
        X_train_flat = load_npy_features_int16(DATA_FILES['X_train'])
        y_train = load_npy_labels_int16(DATA_FILES['y_train'])
        X_test_flat = load_npy_features_int16(DATA_FILES['X_test'])
        y_test = load_npy_labels_int16(DATA_FILES['y_test'])
    except:
        print("Error: Run wine/prepare.py first")
        return
    
    n_features = 13  # 13 wine features (alcohol, malic_acid, ash, etc.)
    n_train = len(y_train)
    n_test = len(y_test)
    
    # Determine number of classes from data
    n_classes = int(max(y_train)) + 1
    
    print(f"Loaded: {n_train} train, {n_test} test samples")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes} (wine cultivars 0, 1, 2)")
    print("Task: Classify wine cultivar")
    
    # Create model
    model = emlearn_extratrees.new(
        n_features, n_classes,
        n_trees=20, max_depth=12, min_samples_leaf=2,
        n_thresholds=15, subsample_ratio=0.8, feature_subsample_ratio=1.0,
        max_nodes=3000, max_samples=500, rng_seed=42
    )
    
    train_start = time.ticks_ms()
    print("Training...")
    model.train(X_train_flat, y_train)
    print(f"Trained: {model.get_n_nodes_used()} nodes")
    train_duration = time.ticks_diff(time.ticks_ms(), train_start)
    print('Time (ms)', train_duration)

    # Test
    correct = 0
    probabilities = array.array('f', [0.0] * n_classes)
    
    # Track class-wise accuracy
    class_correct = [0] * n_classes
    class_total = [0] * n_classes
    
    for i in range(n_test):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        features = array.array('h', X_test_flat[start_idx:end_idx])
        
        predicted = model.predict(features, probabilities)
        actual = y_test[i]
        
        # Track per-class stats
        class_total[actual] += 1
        if predicted == actual:
            correct += 1
            class_correct[actual] += 1
        
        if i < 5:
            conf = max(probabilities)
            print(f"Sample {i}: pred={predicted}, actual={actual}, conf={conf:.3f}")
    
    accuracy = correct / n_test
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{n_test})")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for c in range(n_classes):
        if class_total[c] > 0:
            class_acc = class_correct[c] / class_total[c]
            print(f"  Class {c}: {class_acc:.3f} ({class_correct[c]}/{class_total[c]})")
    
    print(f"Target (sklearn ExtraTrees): ~0.95")
    
    if accuracy >= 0.90:
        print("✅ EXCELLENT: Great wine classification!")
    elif accuracy >= 0.85:
        print("✅ VERY GOOD: Strong wine classification!")
    elif accuracy >= 0.80:
        print("✅ GOOD: Solid wine classification!")
    elif accuracy >= 0.70:
        print("⚠️  FAIR: Working but could improve")
    else:
        print("❌ POOR: Needs significant improvement")

if __name__ == "__main__":
    test_wine()
