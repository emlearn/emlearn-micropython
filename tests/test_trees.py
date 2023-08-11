
import emltrees

import array
import gc

def test_trees_del():
    """
    Deleting the model should free all the memory
    """
    gc.enable()
    gc.collect()
    before_new = gc.mem_alloc()

    model = emltrees.new(5, 30)
    after_new = gc.mem_alloc()
    added = after_new - before_new

    assert added > 100, added
    assert added < 1000, added
    del model
    gc.collect()
    after_del = gc.mem_alloc()
    diff = after_del - before_new
    assert diff == 0, diff

def test_trees_xor():
    """
    Loading simple XOR model and predictions should give correct results
    """

    model = emltrees.new(5, 30)

    # Load a CSV file with the model
    with open('examples/xor_model.csv', 'r') as f:
        emltrees.load_model(model, f)

    # run it
    examples = [
        # input, expected output
        ( [0.0, 0.0], 0 ),
        ( [1.0, 1.0], 0 ),
        ( [0.0, 1.0], 1 ),
        ( [1.0, 0.0], 1 ),
    ]

    for (ex, expect) in examples:
        f = array.array('f', ex)
        result = model.predict(f)
        assert result == expect, (ex, expect, result)

test_trees_xor()
test_trees_del()
