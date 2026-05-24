# Stub file (PEP 484) with API definitions and documentation for native module

"""
Extra Trees (Extremely Randomized Trees) classification.

Tree-based ensemble classifier with randomized splits.
"""

import array
import typing
from typing import Iterator


class Model():
    """An ExtraTrees ensemble model

    Note: Use emlearn_extratrees.new to construct an instance
    """
    def train(self, X : array.array, y : array.array) -> None:
        """
        Train the model on the given data.

        :param X: Training features, int16 array of n_samples * n_features
        :param y: Training labels, int16 array of n_samples
        """
        pass

    def train_init(self, X : array.array, y : array.array) -> None:
        """
        Initialize step-by-step training.

        :param X: Training features, int16 array of n_samples * n_features
        :param y: Training labels, int16 array of n_samples
        """
        pass

    def train_step(self) -> int:
        """
        Process one node of step-by-step training.

        :return: 1 if training is complete, 0 if more steps needed
        """
        pass

    def predict(self, features : array.array, probabilities : array.array) -> int:
        """
        Make a prediction and fill the probabilities buffer.

        :param features: Input features, int16 array of n_features
        :param probabilities: Output buffer, float32 array of n_classes (modified in-place)
        :return: Predicted class index
        """
        pass

    def get_n_features(self) -> int:
        """
        Get the number of features.
        """
        pass

    def get_n_classes(self) -> int:
        """
        Get the number of classes.
        """
        pass

    def get_n_trees(self) -> int:
        """
        Get the number of trees in the ensemble.
        """
        pass

    def get_n_nodes_used(self) -> int:
        """
        Get the number of nodes currently used in the model.
        """
        pass

    def get_n_trees_trained(self) -> int:
        """
        Get the number of trees trained so far.
        """
        pass

    def __del__(self) -> None:
        pass


def new(n_features : int, n_classes : int,
        *, n_trees : int = ..., max_depth : int = ...,
        min_samples_leaf : int = ..., n_thresholds : int = ...,
        subsample_ratio : float = ..., feature_subsample_ratio : float = ...,
        max_nodes : int = ..., max_samples : int = ...,
        rng_seed : int = ..., use_global_feature_range : bool = ...) -> Model:
    """
    Construct a new ExtraTrees model.

    :param n_features: Number of input features
    :param n_classes: Number of output classes
    :param n_trees: Number of trees in the ensemble
    :param max_depth: Maximum tree depth
    :param min_samples_leaf: Minimum samples at a leaf node
    :param n_thresholds: Random thresholds drawn per feature split
    :param subsample_ratio: Fraction of samples used per tree (0.0-1.0)
    :param feature_subsample_ratio: Fraction of features considered per split
    :param max_nodes: Maximum pre-allocated nodes
    :param max_samples: Maximum pre-allocated samples
    :param rng_seed: Random number generator seed
    :param use_global_feature_range: Use global feature range
    """
    pass


def train_steps(model : Model, X : array.array, y : array.array) -> Iterator[int]:
    """
    Generator for step-by-step training.

    Yields the number of trees trained so far after each step.
    Returns when training is complete.

    :param model: ExtraTrees model instance
    :param X: Training features, int16 array of n_samples * n_features
    :param y: Training labels, int16 array of n_samples
    :return: Iterator yielding trees_trained count
    """
    pass
