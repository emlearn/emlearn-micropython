#!/usr/bin/env python3
# Wine Quality test for MicroPython

import array
import gc
import time
import npyfile
import emlearn_extratrees

def load_npy_int16(filename):
    """Load .npy file and convert to int16 array"""
    shape, data = npyfile.load(filename)
    return array.array('h', data)

def test_wine_quality():
    print("=== WINE QUALITY DATASET TEST ===")
    
    # Load preprocessed data
    try:
        X_train_flat = load_npy_int16('X_train.npy')
        y_train = load_npy_int16('y_train.npy')
        X_test_flat = load_npy_int16('X_test.npy')
        y_test = load_npy_int16('y_test.npy')
    except:
        print("Error: Run wine_quality_prep.py first")
        return
    
    n_features = 12  # 11 wine features + wine_type
    n_train = len(y_train)
    n_test = len(y_test)
    
    print(f"Loaded: {n_train} train, {n_test} test samples")
    print(f"Features: {n_features} (alcohol, acidity, etc. + wine_type)")
    print("Task: Predict good wine (quality >= 6) vs poor wine")
    
    # Create model - adjusted for large dataset constraints
    model = emlearn_extratrees.new(
        12,    # n_features
        2,     # n_classes
        5,    # n_trees 
        10,    # max_depth
        3,     # min_samples_leaf
        20,    # n_thresholds
        0.20,  # subsample_ratio (much smaller: 15% of 5197 = ~780 samples)
        0.8,   # feature_subsample_ratio
        2000,  # max_nodes
        10000,  # max_samples (matches subsample size)
        42     # rng_seed
    )
    
    train_start = time.ticks_ms()
    print("Training...")
    model.train(X_train_flat, y_train)
    print(f"Trained: {model.get_n_nodes_used()} nodes")
    train_duration = time.ticks_diff(time.ticks_ms(), train_start)
    print('Time (ms)', train_duration)

    # Test
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    probabilities = array.array('f', [0.0, 0.0])
    
    for i in range(n_test):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        features = array.array('h', X_test_flat[start_idx:end_idx])
        
        predicted = model.predict_proba(features, probabilities)
        actual = y_test[i]
        
        if predicted == actual:
            correct += 1
        
        # Confusion matrix
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1
        
        if i < 5:
            conf = max(probabilities[0], probabilities[1])
            wine_quality = "good" if actual == 1 else "poor"
            pred_quality = "good" if predicted == 1 else "poor"
            print(f"Sample {i}: pred={pred_quality}, actual={wine_quality}, conf={conf:.3f}")
    
    accuracy = correct / n_test
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{n_test})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Target (sklearn): ~0.80")
    
    if accuracy >= 0.78:
        print("✅ EXCELLENT: Great performance on wine quality!")
    elif accuracy >= 0.75:
        print("✅ VERY GOOD: Strong wine classification!")
    elif accuracy >= 0.70:
        print("✅ GOOD: Solid wine quality prediction!")
    elif accuracy >= 0.65:
        print("⚠️  FAIR: Working but could improve")
    else:
        print("❌ POOR: Needs significant improvement")

if __name__ == "__main__":
    test_wine_quality()
