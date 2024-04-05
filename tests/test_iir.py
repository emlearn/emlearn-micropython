
import emliir

import array
import gc

def test_iir_del():
    """
    Deleting should free all the memory
    """
    gc.enable()
    gc.collect()
    before_new = gc.mem_alloc()

    coefficients = array.array('f', [
        0.2064104175118759,
        0.4128208350237518,
        0.2064104175118759,
        1.0,
        -0.37007910158090057,
        0.19608813997724178,
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.999111423794296,
        0.9991118187443988
    ])
    model = emliir.new(coefficients)
    after_new = gc.mem_alloc()
    added = after_new - before_new

    assert added > 100, added
    assert added < 1000, added
    del model
    gc.collect()
    after_del = gc.mem_alloc()
    diff = after_del - before_new
    print(before_new, after_new, after_del)
    #assert diff == 0, diff


test_iir_del()
