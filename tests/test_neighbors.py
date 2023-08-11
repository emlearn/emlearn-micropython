
import array
import emlneighbors

def test_neighbors_trivial():

    n_features = 3
    n_items = 100
    k_neighbors = 1
    m = emlneighbors.open(n_items, n_features, k_neighbors)

    emlneighbors.additem(m, array.array('h', [100, 100, 2]), 1)
    emlneighbors.additem(m, array.array('h', [-100, -100, 2]), 1)

    out = emlneighbors.predict(m, array.array('h', [100, 100, 2]))
    assert out == 1, out

    out = emlneighbors.predict(m, array.array('h', [-100, -100, 2]))
    assert out == 0, out


test_neighbors_trivial()
