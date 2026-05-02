#!/usr/bin/env python3
# MicroPython test script (run with MicroPython after dataset prep)
import array
import gc
import npyfile

DATA_DIR = 'examples/datasets/cancer/'
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
    # Labels are already integers (0.0, 1.0), just convert directly
    labels = [int(v) for v in data]
    return array.array('h', labels)

def test_real_dataset():
    print("=== REAL DATASET TEST ===")
    
    X_train_flat = load_npy_features_int16(DATA_FILES['X_train'])
    y_train = load_npy_labels_int16(DATA_FILES['y_train']) 
    X_test_flat = load_npy_features_int16(DATA_FILES['X_test'])
    y_test = load_npy_labels_int16(DATA_FILES['y_test'])

    
    n_features = 30
    n_train = len(y_train)
    n_test = len(y_test)
    
    print(f"Loaded: {n_train} train, {n_test} test samples")
    print(f"Features: {n_features}")
    
    # Import after data loading to save memory
    import emlearn_extratrees
    
    # Create model
    model = emlearn_extratrees.new(
        30, 2,
        n_trees=20, max_depth=12, min_samples_leaf=2,
        n_thresholds=15, subsample_ratio=0.8, feature_subsample_ratio=0.7,
        max_nodes=3000, max_samples=500, rng_seed=42
    )
    
    print("Training...")
    model.train(X_train_flat, y_train)
    print(f"Trained: {model.get_n_nodes_used()} nodes")
    
    # Test
    correct = 0
    probabilities = array.array('f', [0.0, 0.0])
    
    for i in range(n_test):
        start_idx = i * n_features
        end_idx = start_idx + n_features
        features = array.array('h', X_test_flat[start_idx:end_idx])
        
        predicted = model.predict_proba(features, probabilities)
        actual = y_test[i]
        
        if predicted == actual:
            correct += 1
        
        if i < 5:  # Show first few predictions
            conf = max(probabilities[0], probabilities[1])
            print(f"Sample {i}: pred={predicted}, actual={actual}, conf={conf:.3f}")
    
    accuracy = correct / n_test
    print(f"\nAccuracy: {accuracy:.3f} ({correct}/{n_test})")
    print(f"Target (sklearn): ~0.965")
    
    if accuracy >= 0.90:
        print("✅ EXCELLENT: Matches professional ML performance!")
    elif accuracy >= 0.85:
        print("✅ VERY GOOD: Strong real-world performance!")
    elif accuracy >= 0.80:
        print("✅ GOOD: Solid performance on real data!")
    elif accuracy >= 0.70:
        print("⚠️  FAIR: Working but needs tuning")
    else:
        print("❌ POOR: Significant issues")

if __name__ == "__main__":
    test_real_dataset()
