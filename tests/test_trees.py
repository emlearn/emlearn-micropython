
import emlearn_trees

import array
import gc

def test_trees_del():
    """
    Deleting the model should free all the memory
    """
    gc.enable()
    gc.collect()
    before_new = gc.mem_alloc()

    model = emlearn_trees.new(5, 30, 4)
    after_new = gc.mem_alloc()
    added = after_new - before_new

    assert added > 100, added
    assert added < 1000, added
    del model
    gc.collect()
    after_del = gc.mem_alloc()
    diff = after_del - before_new
    #assert diff == 0, diff

def test_trees_xor():
    """
    Loading simple XOR model and predictions should give correct results
    """

    model = emlearn_trees.new(5, 30, 4)

    # Load a CSV file with the model
    with open('examples/xor_trees/xor_model.csv', 'r') as f:
        emlearn_trees.load_model(model, f)

    # run it
    s = 32767 # max int16
    examples = [
        # input, expected output
        ( [0, 0], 0 ),
        ( [1*s, 1*s], 0 ),
        ( [0, 1*s], 1 ),
        ( [1*s, 0], 1 ),
    ]

    for (ex, expect) in examples:
        f = array.array('h', ex)
        result = model.predict(f)
        assert result == expect, (ex, expect, result)

test_trees_del()
test_trees_xor()

