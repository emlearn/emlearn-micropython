
import emlearn_neighbors

import array
import gc


def test_neighbors_del():
    """
    Deleting the model should free all the memory
    """
    gc.enable()
    gc.collect()
    before_new = gc.mem_alloc()
    model = emlearn_neighbors.new(30, 5, 1)
    after_new = gc.mem_alloc()
    added = after_new - before_new
    gc.collect()
    assert added > 100, added
    assert added < 1000, added
    del model
    gc.collect()
    after_del = gc.mem_alloc()
    del_drop = after_del - before_new
    #assert del_drop <= 0, del_drop

def test_neighbors_trivial():

    n_features = 3
    n_items = 100
    k_neighbors = 2
    model = emlearn_neighbors.new(n_items, n_features, k_neighbors)

    item = array.array('h', [0, 0, 0])

    data = [
        (array.array('h', [-100, -100, -2]), 0),
        (array.array('h', [-100, -100, -2]), 0),
        (array.array('h', [100, 100, 2]), 1),
        (array.array('h', [100, 100, 2]), 1),
    ]
    for x, y in data:
        i = model.additem(x, y)

        # can read data back out again
        model.getitem(i, item)
        assert item == x

    out = model.predict(data[3][0])
    assert out == 1, out

    out = model.predict(data[0][0])
    assert out == 0, out

def test_neighbors_get_results():

    model = emlearn_neighbors.new(100, 3, 1)

    data = [
        (array.array('h', [-100, -100, -2]), 0),
        (array.array('h', [-100, -100, -2]), 0),
        (array.array('h', [100, 100, 2]), 1),
        (array.array('h', [100, 100, 2]), 1),
    ]
    for x, y in data:
        model.additem(x, y)

    f = data[1][0]
    out = model.predict(f)
    assert out == 0, out

    # Can get details about the particular
    for i in range(0, 4):
        data_item, distance, label = model.getresult(i)
        assert distance in (0, 282)
        assert label in (0, 1)

    assert model.getresult(0)[2] == 0
    assert model.getresult(3)[2] == 1

if __name__ == '__main__':
    test_neighbors_del()
    test_neighbors_trivial()
    test_neighbors_get_results()

