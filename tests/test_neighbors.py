
import array
import emlneighbors

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


test_neighbors_trivial()
