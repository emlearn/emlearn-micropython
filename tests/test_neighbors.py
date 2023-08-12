
import emlneighbors

import array
import gc


def test_neighbors_del():
    """
    Deleting the model should free all the memory
    """
    gc.enable()
    gc.collect()
    before_new = gc.mem_alloc()

    model = emlneighbors.new(30, 5, 1)
    after_new = gc.mem_alloc()
    added = after_new - before_new

    assert added > 100, added
    assert added < 1000, added
    del model
    gc.collect()
    after_del = gc.mem_alloc()
    diff = after_del - before_new
    assert diff == 0, diff

def test_neighbors_trivial():

    n_features = 3
    n_items = 100
    k_neighbors = 2
    model = emlneighbors.new(n_items, n_features, k_neighbors)

    data = [
        (array.array('h', [-100, -100, -2]), 0),
        (array.array('h', [-100, -100, -2]), 0),
        (array.array('h', [100, 100, 2]), 1),
        (array.array('h', [100, 100, 2]), 1),
    ]
    for x, y in data:
        model.additem(x, y)

    out = model.predict(data[3][0])
    assert out == 1, out

    out = model.predict(data[0][0])
    assert out == 0, out

def test_neighbors_get_results():

    model = emlneighbors.new(100, 3, 1)

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

    for i in range(0, 4):
        data_item, distance, label = model.getresult(i)
        assert distance in (0, 282)
        assert label in (0, 1)

    assert model.getresult(0)[2] == 0
    assert model.getresult(3)[2] == 1

test_neighbors_trivial()
test_neighbors_del()
test_neighbors_get_results()
