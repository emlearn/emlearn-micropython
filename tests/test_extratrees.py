# Debug the prediction logic specifically
import array
import emlearn_extratrees


def argmax(arr, n):
    best_idx = 0
    best_val = arr[0]
    for i in range(1, n):
        if arr[i] > best_val:
            best_val = arr[i]
            best_idx = i
    return best_idx

def test_single_tree_prediction():
    """Test with just one tree to isolate prediction issues"""
    print("=== Single Tree Prediction Debug ===")
    
    # Extremely simple case: one tree, clear split
    X = array.array('h', [
        0, 0,      # Class 0
        1000, 1000, # Class 1
    ])
    y = array.array('h', [0, 1])
    
    print("Data: (0,0)->0, (1000,1000)->1")
    
    # Single tree with enough depth and thresholds
    model = emlearn_extratrees.new(2, 2, n_trees=1, max_depth=5, n_thresholds=100)
    model.train(X, y)
    
    nodes_used = model.get_n_nodes_used()
    print("Nodes used: {}".format(nodes_used))
    
    if nodes_used > 1:
        print("✓ Tree has internal nodes")
        
        # Test both training samples
        probabilities = array.array('f', [0.0, 0.0])
        
        for i in range(2):
            x1, x2 = X[i*2], X[i*2+1]
            expected = y[i]
            
            test_features = array.array('h', [x1, x2])
            predicted = model.predict(test_features, probabilities)
            
            print("  ({}, {}) -> pred={}, exp={}, probs=[{:.3f}, {:.3f}] {}".format(
                x1, x2, predicted, expected,
                probabilities[0], probabilities[1],
                "✓" if predicted == expected else "✗"))
    else:
        print("✗ Tree is just a leaf")

def test_class_bias():
    """Test if there's a systematic bias toward class 0"""
    print("\n=== Class Bias Debug ===")
    
    # Test with data that should heavily favor class 1
    X = array.array('h', [
        # Only 1 sample of class 0
        0, 0,
        
        # Many samples of class 1 
        100, 100,
        200, 200, 
        300, 300,
        400, 400,
        500, 500,
    ])
    
    y = array.array('h', [0, 1, 1, 1, 1, 1])
    
    print("Data: 1 sample class 0, 5 samples class 1")
    
    model = emlearn_extratrees.new(2, 2, n_trees=3, max_depth=5, n_thresholds=50)
    model.train(X, y)
    
    # Test on a clear class 1 example
    test_features = array.array('h', [300, 300])
    probabilities = array.array('f', [0.0, 0.0])
    model.predict(test_features, probabilities)
    predicted = argmax(probabilities, 2)
    
    print("Prediction for (300,300): {} (should strongly favor class 1)".format(predicted))
    print("Probabilities: [{:.3f}, {:.3f}]".format(probabilities[0], probabilities[1]))

def test_manual_verification():
    """Manually verify what the model should predict"""
    print("\n=== Manual Verification ===")
    
    # Create a dataset where we can manually verify the correct answer
    X = array.array('h', [
        # Group A: x1 <= 50, all class 0
        10, 999,
        20, 888,
        30, 777,
        
        # Group B: x1 > 50, all class 1  
        60, 111,
        70, 222,
        80, 333,
    ])
    
    y = array.array('h', [0, 0, 0, 1, 1, 1])
    
    print("Perfect split rule: x1 <= 50 -> class 0, x1 > 50 -> class 1")
    print("Training data:")
    for i in range(6):
        x1, x2 = X[i*2], X[i*2+1]
        label = y[i]
        print("  ({}, {}) -> {}".format(x1, x2, label))
    
    # Train with parameters that should definitely work
    model = emlearn_extratrees.new(2, 2, n_trees=10, max_depth=10, n_thresholds=100)
    model.train(X, y)
    
    print("\nNodes used: {}".format(model.get_n_nodes_used()))
    
    # Test the rule
    test_cases = [
        (25, 0),   # x1=25 <= 50, should be class 0
        (75, 1),   # x1=75 > 50, should be class 1
        (40, 0),   # x1=40 <= 50, should be class 0
        (90, 1),   # x1=90 > 50, should be class 1
    ]
    
    print("\nTesting the learned rule:")
    probabilities = array.array('f', [0.0, 0.0])
    
    all_correct = True
    for x1_val, expected in test_cases:
        test_features = array.array('h', [x1_val, 500])  # x2 irrelevant
        predicted = model.predict(test_features, probabilities)
        
        is_correct = predicted == expected
        if not is_correct:
            all_correct = False
            
        print("  x1={} -> pred={}, exp={}, probs=[{:.3f}, {:.3f}] {}".format(
            x1_val, predicted, expected,
            probabilities[0], probabilities[1],
            "✓" if is_correct else "✗"))
    
    if all_correct:
        print("✓ All predictions correct - model is working!")
    else:
        print("✗ Model is not learning the simple rule")
        
        # Additional diagnosis
        print("\nDiagnosis:")
        if probabilities[0] == 1.0 and probabilities[1] == 0.0:
            print("- All trees voting for class 0 -> prediction logic issue")
        elif probabilities[0] == 0.0 and probabilities[1] == 1.0:
            print("- All trees voting for class 1 -> prediction logic issue")
        else:
            print("- Mixed votes but wrong outcome -> majority vote logic issue")

def test_train_step_by_step():
    """Test incremental training using train_init/train_step"""
    print("\n=== Step-by-step Training ===")
    
    X = array.array('h', [
        0, 0,
        100, 100,
        200, 200,
        300, 300,
    ])
    y = array.array('h', [0, 0, 1, 1])
    
    # 3 trees
    model = emlearn_extratrees.new(2, 2, n_trees=3, max_depth=5, n_thresholds=50)
    model.train_init(X, y)
    
    steps = 0
    trees_done_list = []
    while True:
        done = model.train_step()
        steps += 1
        trees_done = model.get_n_trees_trained()
        trees_done_list.append(trees_done)
        if done:
            break
    
    print("Steps taken: {}".format(steps))
    print("Trees trained: {}".format(model.get_n_trees_trained()))
    print("Nodes used: {}".format(model.get_n_nodes_used()))
    
    # Verify predictions still work
    probabilities = array.array('f', [0.0, 0.0])
    test_features = array.array('h', [0, 0])
    predicted = model.predict(test_features, probabilities)
    print("Predict (0,0): {} probs=[{:.3f}, {:.3f}]".format(predicted, probabilities[0], probabilities[1]))
    assert predicted == 0, "Expected class 0"
    
    test_features = array.array('h', [300, 300])
    predicted = model.predict(test_features, probabilities)
    print("Predict (300,300): {} probs=[{:.3f}, {:.3f}]".format(predicted, probabilities[0], probabilities[1]))
    assert predicted == 1, "Expected class 1"
    
    print("✓ Step-by-step training works")

def test_train_generator():
    """Test training using the Python generator"""
    print("\n=== Generator Training ===")
    
    X = array.array('h', [
        0, 0,
        100, 100,
        200, 200,
        300, 300,
    ])
    y = array.array('h', [0, 0, 1, 1])
    
    model = emlearn_extratrees.new(2, 2, n_trees=5, max_depth=5, n_thresholds=50)
    
    trees_progress = []
    for trees_done in emlearn_extratrees.train_steps(model, X, y):
        trees_progress.append(trees_done)
    
    print("Progress: {}".format(trees_progress))
    print("Trees trained: {}".format(model.get_n_trees_trained()))
    print("Nodes used: {}".format(model.get_n_nodes_used()))
    
    assert model.get_n_trees_trained() == 5
    
    # Verify predictions
    probabilities = array.array('f', [0.0, 0.0])
    test_features = array.array('h', [0, 0])
    model.predict(test_features, probabilities)
    predicted = argmax(probabilities, 2)
    assert predicted == 0, "Expected class 0"
    
    test_features = array.array('h', [300, 300])
    model.predict(test_features, probabilities)
    predicted = argmax(probabilities, 2)
    assert predicted == 1, "Expected class 1"
    
    print("✓ Generator training works")

def test_train_step_same_as_train():
    """Verify step-by-step training produces same result as bulk train"""
    print("\n=== Step vs Bulk Equivalence ===")
    
    X = array.array('h', [
        0, 0,
        50, 50,
        100, 100,
        150, 150,
    ])
    y = array.array('h', [0, 0, 1, 1])
    
    # Bulk train
    model_bulk = emlearn_extratrees.new(2, 2, n_trees=3, max_depth=5, n_thresholds=50, max_samples=100, rng_seed=42)
    model_bulk.train(X, y)
    
    # Step-by-step train
    model_step = emlearn_extratrees.new(2, 2, n_trees=3, max_depth=5, n_thresholds=50, max_samples=100, rng_seed=42)
    model_step.train_init(X, y)
    while not model_step.train_step():
        pass
    
    nodes_bulk = model_bulk.get_n_nodes_used()
    nodes_step = model_step.get_n_nodes_used()
    print("Bulk nodes: {}, Step nodes: {}".format(nodes_bulk, nodes_step))
    
    assert nodes_bulk == nodes_step, "Step-by-step should produce same number of nodes"
    assert model_step.get_n_trees_trained() == 3
    
    print("✓ Step and bulk produce identical results")

if __name__ == "__main__":
    print("Prediction Logic Debug")
    print("=" * 50)
    
    try:
        test_single_tree_prediction()
        test_class_bias()
        test_manual_verification()
        test_train_step_by_step()
        test_train_generator()
        test_train_step_same_as_train()
        
        print("\n" + "="*50)
        print("Prediction debug completed!")
        
    except Exception as e:
        print("Error during debugging: {}".format(e))
        import sys
        sys.print_exception(e)
