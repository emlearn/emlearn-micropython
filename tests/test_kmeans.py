
import emlkmeans

import array
import gc

def test_kmeans_two_clusters():
    """
    Data that is grouped into high/low should be clusterable into 2
    """

    n_features = 3
    dataset = array.array('B', [
        0, 0, 0,
        10, 5, 2,

        200, 50, 100,
        255, 255, 255,        
    ])

    centroids = array.array('B', [
        0, 0, 0,
        200, 200, 200,
    ])

    assignments = emlkmeans.cluster(dataset, centroids, channels=n_features)
    assert len(assignments) == len(dataset)/n_features
    assert list(assignments) == [0, 0, 1, 1], assignments


test_kmeans_two_clusters()
