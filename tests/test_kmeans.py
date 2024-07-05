
import emlkmeans

import array
import gc

def make_two_cluster_data(typecode):

    n_features = 3
    dataset = [
        # cluster1
        0, 0, 0,
        10, 5, 2,
    
        # cluster2
        200, 50, 100,
        255, 255, 255,        
    ]

    centroids = [
        0, 0, 0,
        200, 200, 200,
    ]

    if typecode == 'bytearray':
        dataset = bytearray(dataset)
        centroids = bytearray(centroids)        
    else:
        dataset = array.array(typecode, dataset)
        centroids = array.array(typecode, centroids)

    return dataset, centroids


def test_kmeans_two_clusters():
    """
    Data that is grouped into high/low should be clusterable into 2
    """

    n_features = 3
    # test both with "bytearray" and "array.array"
    for typecode in ['bytearray', 'B']:
        dataset, centroids = make_two_cluster_data(typecode)

        assignments = emlkmeans.cluster(dataset, centroids, channels=n_features)
        assert len(assignments) == len(dataset)/n_features
        assert list(assignments) == [0, 0, 1, 1], assignments


test_kmeans_two_clusters()
