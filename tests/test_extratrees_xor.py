# Final comprehensive XOR test suite
import array
import emlearn_extratrees

def test_xor_comprehensive():
    """Comprehensive XOR test with the fixed algorithm"""
    print("=== Comprehensive XOR Test ===")
    
    # XOR training data - repeated for better training
    base_pattern = [
        (0, 0, 0),     # XOR: (0,0) -> 0
        (0, 100, 1),   # XOR: (0,1) -> 1  
        (100, 0, 1),   # XOR: (1,0) -> 1
        (100, 100, 0), # XOR: (1,1) -> 0
    ]
    
    # Repeat pattern multiple times to give ensemble more training data
    X_data = []
    y_data = []
    for _ in range(8):  # 32 samples total
        for x1, x2, y in base_pattern:
            X_data.extend([x1, x2])
            y_data.append(y)
    
    X = array.array('h', X_data)
    y = array.array('h', y_data)
    
    print(f"Training data: {len(y_data)} samples (8x XOR pattern)")
    
    # Test with ensemble of trees (now that individual trees work)
    model = emlearn_extratrees.new(
        2,      # n_features
        2,      # n_classes  
        10,     # n_trees (ensemble)
        8,      # max_depth
        1,      # min_samples_leaf
        10,     # n_thresholds
        0.8,    # subsample_ratio (80% for diversity)
        1.0,    # feature_subsample_ratio (use both features)
        500,    # max_nodes
        100,    # max_samples
        42      # rng_seed
    )
    
    model.train(X, y)
    
    print(f"Model: {model.get_n_trees()} trees, {model.get_n_nodes_used()} nodes total")
    
    # Test core XOR patterns
    test_cases = [
        ([0, 0], 0),
        ([0, 100], 1), 
        ([100, 0], 1),
        ([100, 100], 0),
    ]
    
    print("\nCore XOR Results:")
    correct = 0
    probabilities = array.array('f', [0.0, 0.0])
    
    for features, expected in test_cases:
        test_features = array.array('h', features)
        predicted = model.predict_proba(test_features, probabilities)
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        confidence = max(probabilities[0], probabilities[1])
        print(f"  {features} -> pred={predicted}, exp={expected}, conf={confidence:.2f} {'✓' if is_correct else '✗'}")
    
    core_accuracy = 100.0 * correct / 4
    print(f"Core XOR Accuracy: {core_accuracy:.0f}%")
    
    # Test interpolation (intermediate values)
    print("\nInterpolation Test:")
    interpolation_cases = [
        ([25, 25], "?"),   # Between (0,0) and (100,100) - ambiguous
        ([25, 75], "?"),   # Between (0,100) and (100,0) - ambiguous  
        ([10, 90], 1),     # Closer to (0,100) -> should be 1
        ([90, 10], 1),     # Closer to (100,0) -> should be 1
        ([90, 90], 0),     # Closer to (100,100) -> should be 0
        ([10, 10], 0),     # Closer to (0,0) -> should be 0
    ]
    
    for features, expected in interpolation_cases:
        test_features = array.array('h', features)
        predicted = model.predict_proba(test_features, probabilities)
        confidence = max(probabilities[0], probabilities[1])
        
        if expected == "?":
            marker = "?"
        else:
            marker = "✓" if predicted == expected else "✗"
            
        print(f"  {features} -> pred={predicted}, exp={expected}, conf={confidence:.2f} {marker}")
    
    return core_accuracy >= 100

def test_xor_robustness():
    """Test XOR robustness with different parameters"""
    print("\n=== XOR Robustness Test ===")
    
    # XOR data
    X_data = [0, 0, 0, 100, 100, 0, 100, 100] * 6  # 24 samples
    y_data = [0, 1, 1, 0] * 6
    
    X = array.array('h', X_data)
    y = array.array('h', y_data)
    
    configs = [
        (5, 6, "5 trees, depth 6"),
        (15, 10, "15 trees, depth 10"),
        (20, 12, "20 trees, depth 12"),
    ]
    
    results = []
    
    for n_trees, max_depth, desc in configs:
        print(f"\nTesting {desc}:")
        
        model = emlearn_extratrees.new(2, 2, n_trees, max_depth, 1, 8, 0.9, 1.0, 1000, 100, 123)
        model.train(X, y)
        
        # Test all XOR cases
        correct = 0
        probabilities = array.array('f', [0.0, 0.0])
        test_cases = [([0, 0], 0), ([0, 100], 1), ([100, 0], 1), ([100, 100], 0)]
        
        for features, expected in test_cases:
            test_features = array.array('h', features)
            predicted = model.predict_proba(test_features, probabilities)
            if predicted == expected:
                correct += 1
        
        accuracy = 100.0 * correct / 4
        results.append(accuracy)
        print(f"  Accuracy: {accuracy:.0f}% ({correct}/4 correct)")
    
    avg_accuracy = sum(results) / len(results)
    print(f"\nAverage accuracy across configs: {avg_accuracy:.0f}%")
    
    return avg_accuracy >= 75

def test_xor_different_values():
    """Test XOR with different value ranges"""
    print("\n=== XOR with Different Value Ranges ===")
    
    # Test with different value ranges to ensure generalization
    test_ranges = [
        ([0, 1], "Binary"),
        ([0, 10], "0-10"),
        ([0, 1000], "0-1000"),
        ([-50, 50], "-50 to 50"),
    ]
    
    results = []
    
    for value_range, desc in test_ranges:
        print(f"\nTesting {desc} range:")
        
        low, high = value_range
        X_data = [
            low, low,      # (low,low) -> 0
            low, high,     # (low,high) -> 1
            high, low,     # (high,low) -> 1  
            high, high,    # (high,high) -> 0
        ] * 8  # 32 samples
        y_data = [0, 1, 1, 0] * 8
        
        X = array.array('h', X_data)
        y = array.array('h', y_data)
        
        model = emlearn_extratrees.new(2, 2, 12, 10, 1, 10, 0.8, 1.0, 800, 100, 456)
        model.train(X, y)
        
        # Test
        test_cases = [
            ([low, low], 0),
            ([low, high], 1), 
            ([high, low], 1),
            ([high, high], 0),
        ]
        
        correct = 0
        probabilities = array.array('f', [0.0, 0.0])
        
        for features, expected in test_cases:
            test_features = array.array('h', features)
            predicted = model.predict_proba(test_features, probabilities)
            if predicted == expected:
                correct += 1
        
        accuracy = 100.0 * correct / 4
        results.append(accuracy)
        print(f"  Accuracy: {accuracy:.0f}%")
    
    avg_accuracy = sum(results) / len(results)
    print(f"\nAverage across value ranges: {avg_accuracy:.0f}%")
    
    return avg_accuracy >= 75

if __name__ == "__main__":
    print("🔥 FIXED XOR TEST SUITE 🔥")
    print("=" * 60)
    
    try:
        # Test 1: Comprehensive XOR
        success1 = test_xor_comprehensive()
        
        if success1:
            print("\n✅ COMPREHENSIVE XOR TEST PASSED!")
            
            # Test 2: Robustness
            success2 = test_xor_robustness()
            
            # Test 3: Different value ranges  
            success3 = test_xor_different_values()
            
            if success2:
                print("\n✅ ROBUSTNESS TEST PASSED!")
            if success3:
                print("\n✅ VALUE RANGE TEST PASSED!")
            
            if success1 and success2 and success3:
                print("\n🎉🎉🎉 ALL XOR TESTS PASSED! 🎉🎉🎉")
                print("Your Extra Trees implementation is WORKING PERFECTLY!")
                print("The algorithm can now learn complex non-linear patterns like XOR!")
            else:
                print("\n🔥 Core XOR works! Some edge cases may need fine-tuning.")
                
        else:
            print("\n❌ Something is still wrong with the core algorithm")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import sys
        sys.print_exception(e)
    
    print("\n" + "="*60)
