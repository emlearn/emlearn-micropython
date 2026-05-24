# Stub file (PEP 484) with API definitions and documentation for native module

"""
Partial Least Squares Regression (PLSR)

Implemented using *eml_plsr* from the emlearn C library (https://github.com/emlearn/emlearn).
"""

import array
import typing


class Model():
    """A PLSR model

    Note: Use emlearn_plsr.new to construct an instance
    """
    def fit_start(self, X : array.array, y : array.array) -> None:
        """
        Start iterative fitting of the model.

        :param X: Training input data, float32 array of n_samples * n_features
        :param y: Training target data, float32 array of n_samples
        """
        pass

    def step(self, tolerance : float = ...) -> None:
        """
        Perform one NIPALS iteration step for the current component.

        :param tolerance: Convergence tolerance (optional)
        """
        pass

    def finalize_component(self) -> None:
        """
        Finalize the current component and prepare for the next one.
        """
        pass

    def is_converged(self) -> bool:
        """
        Check if the current component has converged.
        """
        pass

    def is_complete(self) -> bool:
        """
        Check if all components have been trained.
        """
        pass

    def predict(self, x : array.array) -> float:
        """
        Predict the target value for a single sample.

        :param x: Input features, float32 array of n_features
        :return: Predicted target value
        """
        pass

    def get_convergence_metric(self) -> float:
        """
        Get the convergence metric for the current component.
        """
        pass

    def set_auto_center(self, value : bool) -> None:
        """
        Enable or disable automatic centering of data during fitting.

        :param value: Whether to auto-center
        """
        pass

    def get_auto_center(self) -> bool:
        """
        Get the auto-centering flag.
        """
        pass

    def __del__(self) -> None:
        pass


def new(n_samples : int, n_features : int, n_components : int) -> Model:
    """
    Construct a new PLSR model.

    :param n_samples: Number of training samples
    :param n_features: Number of input features
    :param n_components: Number of PLS components
    """
    pass


def fit(model : Model, X_train : array.array, y_train : array.array,
        max_iterations : int = ...,
        tolerance : float = ...,
        check_interval : int = ...,
        verbose : int = ...) -> typing.Tuple[int, float]:
    """
    Train the PLSR model using a NIPALS iterative fitting loop.

    :param model: PLSR model instance
    :param X_train: Training input data, float32 array of n_samples * n_features
    :param y_train: Training target data, float32 array of n_samples
    :param max_iterations: Maximum iterations per component
    :param tolerance: Convergence tolerance
    :param check_interval: Check convergence every N iterations
    :param verbose: Verbosity level
    :return: Tuple of (total_iterations, final_convergence_metric)
    """
    pass
