
import array

# FIXME: get Exception: can't merge files when more than one contains native code
# when the native code emitter is enabled. Which is critical for performance...
# maybe we can move the inner part into a kmeans_cluster_step done in C

#@micropython.native
def cluster(values, centroids,
        channels=3, max_iter=10, stop_changes=0):
    """
    Perform K-Means clustering of @values

    Uses the @centroid as initial values for the clusters.

    NOTE: will mutate @centroids
    """

    n_clusters = len(centroids) // channels
    n_samples = len(values) // channels

    assert channels == 3, 'only support 3 channels for now'

    assert channels < 255, channels
    assert n_clusters < 255, n_clusters
    assert n_samples < 65535, n_samples

    assignments = array.array('B', (255 for _ in range(n_samples)))
    cluster_sums = array.array('L', (0 for _ in range(n_clusters*channels)))
    cluster_counts = array.array('H', (0 for _ in range(n_clusters)))

    for i in range(max_iter):

        ## update sample assignments
        changes = 0
        for s in range(n_samples):
            v = values[(s*channels)+0:(s*channels)+3]
            #v0 = values[(s*channels)+0]
            #v1 = values[(s*channels)+1]
            #v2 = values[(s*channels)+2]

            idx, dist = euclidean_argmin(centroids, v)
            #idx, dist = 0, 0

            if idx != assignments[s]:
                changes += 1            
            assignments[s] = idx

        print('iter', i, changes)
        if changes <= stop_changes:
            break

        ## update cluster centroids
        # reset old values
        for c in range(n_clusters*channels):
            cluster_sums[c] = 0
        for c in range(n_clusters):
            cluster_counts[c] = 0

        # evaluate samples
        for s in range(n_samples):
            c = assignments[s]
            v0 = values[(s*channels)+0]
            v1 = values[(s*channels)+1]
            v2 = values[(s*channels)+2]            

            cluster_sums[(c*channels)+0] += v0
            cluster_sums[(c*channels)+1] += v1
            cluster_sums[(c*channels)+2] += v2
            cluster_counts[c] += 1

        # set new centroid means
        for c in range(n_clusters):
            count = cluster_counts[c]
            if count == 0:
                continue

            centroids[(c*channels)+0] = cluster_sums[(c*channels)+0] // count
            centroids[(c*channels)+1] = cluster_sums[(c*channels)+1] // count
            centroids[(c*channels)+2] = cluster_sums[(c*channels)+2] // count

        #yield assignments
        # TODO: make this into a generator? so other work can be done in between
        

    return assignments
