
import emlkmeans

import array
import gc

def test_kmeans_simple():
    """
    Running should give output
    """

    n_features = 3
    dataset = array.array('B', [
        
    ])

    centroids = array.array('B', [

    ])

    assignments = emlkmeans.cluster(dataset, centroids, channels=n_features)
    assert len(assignments) == len(dataset / n_features)


test_kmeans_simple()
