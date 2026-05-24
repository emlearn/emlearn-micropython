# Stub file (PEP 484) with API definitions and documentation for native module

"""
K-Means clustering with C-accelerated distance computation.
"""

import array
import typing
from typing import Iterator


def euclidean_argmin(vectors : array.array, point : array.array) -> typing.Tuple[int, int]:
    """
    Find the closest centroid/vector to a given point.

    :param vectors: All vectors concatenated, uint8 array of n_vectors * n_channels
    :param point: Query point, uint8 array of n_channels
    :return: Tuple of (closest_vector_index, squared_distance)
    """
    pass


def cluster_iter(values : array.array, centroids : array.array,
                 assignments : array.array, features : int,
                 max_iter : int = ..., stop_changes : int = ...) -> Iterator[int]:
    """
    Perform K-Means clustering with iteration yielding.

    :param values: Input data, uint8 array of n_samples * features
    :param centroids: Initial centroids, uint8 array of n_clusters * features (modified in-place)
    :param assignments: Output assignments, uint8 array of n_samples (modified in-place)
    :param features: Number of features
    :param max_iter: Maximum number of iterations
    :param stop_changes: Stop if changes in assignments drop below this count
    :return: Iterator yielding number of assignment changes per iteration
    """
    pass


def cluster(values : array.array, centroids : array.array,
            features : int, **kwargs) -> array.array:
    """
    Run K-Means clustering and return sample assignments.

    :param values: Input data, uint8 array of n_samples * features
    :param centroids: Initial centroids, uint8 array of n_clusters * features
    :param features: Number of features
    :param kwargs: Additional arguments (max_iter, stop_changes)
    :return: Array of cluster assignments for each sample
    """
    pass
