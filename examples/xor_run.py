# device/micropython code

import emltrees
import array

def test_trees_del():
    import gc
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
    assert after_del == before_new

test_trees_del()

model = emltrees.new(5, 30)

# Load a CSV file with the model
with open('xor_model.csv', 'r') as f:
    emltrees.load_model(model, f)

# run it
examples = [
    array.array('f', [0.0, 0.0]),
    array.array('f', [1.0, 1.0]),
    array.array('f', [0.0, 1.0]),
    array.array('f', [1.0, 0.0]),
]

for ex in examples: 
    result = model.predict(ex)
    print(ex, result)

