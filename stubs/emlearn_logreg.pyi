# Stub file (PEP 484) with API definitions and documentation for native module

"""
Logistic Regression (binary and multiclass).

For more complicated training needs, use the `train` or `train_batches` helper functions.
"""

import array
import typing


class Model():
    """A logistic regression model

    Note: Use emlearn_logreg.new to construct an instance
    """
    def predict(self, features : array.array, probs : array.array, logits : array.array) -> int:
        """
        Run prediction and return the predicted class index.

        :param features: Input features, float32 array of n_features
        :param probs: Output buffer for probabilities, float32 array of n_classes (modified in-place)
        :param logits: Output buffer for logits, float32 array of n_classes (modified in-place)
        :return: Index of the predicted class
        """
        pass

    def step(self, X : array.array, y : array.array,
             logits : array.array, probs : array.array,
             bias_grads : array.array) -> None:
        """
        Perform a single training iteration.

        :param X: Batch features, float32 array of n_samples * n_features
        :param y: Batch targets (one-hot), float32 array of n_samples * n_classes
        :param logits: Workspace buffer, float32 array of n_classes
        :param probs: Workspace buffer, float32 array of n_classes
        :param bias_grads: Workspace buffer, float32 array of n_classes
        """
        pass

    def get_weights(self, out : array.array) -> None:
        """
        Copy the model weights into an output buffer.

        :param out: Float32 buffer of n_features * n_classes (modified in-place)
        """
        pass

    def set_weights(self, weights : array.array) -> None:
        """
        Set the model weights from a buffer.

        :param weights: Float32 array of n_features * n_classes
        """
        pass

    def get_bias(self, out : array.array) -> None:
        """
        Copy the model biases into an output buffer.

        :param out: Float32 buffer of n_classes (modified in-place)
        """
        pass

    def set_bias(self, bias : array.array) -> None:
        """
        Set the model biases.

        :param bias: Float32 array of n_classes
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

    def score_logloss(self, X : array.array, y : array.array,
                      logits : array.array, probs : array.array) -> float:
        """
        Compute the log-loss on a dataset.

        :param X: Features, float32 array of n_samples * n_features
        :param y: Targets (one-hot), float32 array of n_samples * n_classes
        :param logits: Workspace buffer, float32 array of n_classes
        :param probs: Workspace buffer, float32 array of n_classes
        :return: The log-loss value
        """
        pass

    def __del__(self) -> None:
        pass


def new(n_features : int, n_classes : int,
        learning_rate : float, lambda_l2 : float, lambda_l1 : float) -> Model:
    """
    Construct a new logistic regression model.

    :param n_features: Number of input features
    :param n_classes: Number of output classes
    :param learning_rate: Learning rate for gradient descent
    :param lambda_l2: L2 regularization strength
    :param lambda_l1: L1 regularization strength
    """
    pass


def train(model : Model, X_train : array.array, y_train : array.array,
          max_iterations : int = ...,
          tolerance : float = ...,
          check_interval : int = ...,
          divergence_factor : float = ...,
          score_limit : typing.Optional[float] = ...,
          verbose : int = ...) -> typing.Tuple[int, float]:
    """
    Full-dataset training loop for logistic regression.

    :param model: Logistic regression model instance
    :param X_train: Training features, float32 array of n_samples * n_features
    :param y_train: Training targets (one-hot), float32 array of n_samples * n_classes
    :param max_iterations: Maximum number of training steps
    :param tolerance: Convergence tolerance
    :param check_interval: Check convergence every N iterations
    :param divergence_factor: Divergence detection factor
    :param score_limit: Stop training if score reaches this value
    :param verbose: Verbosity level
    :return: Tuple of (iterations_completed, final_loss)
    """
    pass


def train_batches(model : Model,
                   batch_iter_factory : typing.Callable[[], typing.Iterator[typing.Tuple[array.array, array.array]]],
                   max_iterations : int = ...,
                   tolerance : float = ...,
                   check_interval : int = ...,
                   divergence_factor : float = ...,
                   score_limit : typing.Optional[float] = ...,
                   verbose : int = ...,
                   score_batches : typing.Optional[typing.Callable[[Model], float]] = ...) -> typing.Tuple[int, float]:
    """
    Train logistic regression model using externally provided batches.

    :param model: Logistic regression model instance
    :param batch_iter_factory: Callable returning a fresh iterator over (X_batch, y_batch) tuples
    :param max_iterations: Maximum number of training epochs
    :param tolerance: Convergence tolerance
    :param check_interval: Check convergence every N epochs
    :param divergence_factor: Divergence detection factor
    :param score_limit: Stop training if score reaches this value
    :param verbose: Verbosity level
    :param score_batches: Optional callable computing score from the model
    :return: Tuple of (iterations_completed, final_loss)
    """
    pass
