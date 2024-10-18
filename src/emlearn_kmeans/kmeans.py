
import array

# FIXME: get Exception: can't merge files when more than one contains native code
# when the native code emitter is enabled. Which is critical for performance...
# maybe we can move the inner part into a kmeans_cluster_step done in C


#@micropython.native
def cluster_iter(values, centroids, assignments, features,
        max_iter=10, stop_changes=0):
    """
    Perform K-Means clustering of @values

    Uses the @centroid as initial values for the clusters.

    NOTE: will mutate @centroids
    """

    channels = features
    n_clusters = len(centroids) // channels
    n_samples = len(values) // channels

    assert channels < 255, channels
    assert n_clusters < 255, n_clusters
    assert n_samples < 65535, n_samples

    cluster_sums = array.array('L', (0 for _ in range(n_clusters*channels)))
    cluster_counts = array.array('H', (0 for _ in range(n_clusters)))

    for i in range(max_iter):

        ## update sample assignments
        changes = 0
        for s in range(n_samples):
            v = values[s*channels:(s+1)*channels]

            # PERF: considering taking all N points at the same time, filling indices and (optionally) distances
            idx, dist = euclidean_argmin(centroids, v)
            #idx, dist = 0, 0

            if idx != assignments[s]:
                changes += 1            
            assignments[s] = idx

        # Pass control back to caller
        # So one can do other work between the iterations
        # or implement custom stopping criteria
        yield changes

        if changes <= stop_changes:
            break

        ## update cluster centroids
        # PERF: consider moving this to C. With a update_centroids() function
        # reset old values
        for c in range(n_clusters*channels):
            cluster_sums[c] = 0
        for c in range(n_clusters):
            cluster_counts[c] = 0

        # evaluate samples
        for s in range(n_samples):
            c = assignments[s]
            for i in range(channels):
                cluster_sums[(c*channels)+i] += values[(s*channels)+i]

            cluster_counts[c] += 1

        # set new centroid means
        for c in range(n_clusters):
            count = cluster_counts[c]
            if count == 0:
                continue

            for i in range(channels):
                centroids[(c*channels)+i] = cluster_sums[(c*channels)+i] // count
        


def cluster(values, centroids, features, **kwargs):
    """Convenience wrapper around cluster_iter"""

    n_samples = len(values) // features
    assignments = array.array('B', (255 for _ in range(n_samples)))

    generator = cluster_iter(values, centroids, assignments, features, **kwargs)
    for changes in generator:
        pass

    return assignments

