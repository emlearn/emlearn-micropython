# Stub file (PEP 484) with API definitions and documentation for native module

"""
SEFR linear classifier (binary and multiclass), trained in one pass without iterations.

Training keeps per-class feature sums, class counts and the min and max of each feature,
so data can be streamed in batches with `Model.partial_fit`.
An optional second pass (`Model.partial_fit_lda`) fits linear discriminant analysis
on the class scores, and folds the result into the same weights.

For training on a full dataset in memory, use the `train` helper function.
"""

import array


class Model():
    """A SEFR model

    Note: Use emlearn_sefr.new to construct an instance
    """
    def partial_fit(self, X : array.array, y : array.array) -> None:
        """
        Add a batch of samples to the model statistics.

        :param X: Batch features, float32 array of n_samples * n_features
        :param y: Batch targets as class indices, float32 array of n_samples
        """
        pass

    def finalize(self) -> None:
        """
        Compute the weights and biases from all samples seen so far.
        Can be called again after more calls to partial_fit.
        """
        pass

    def partial_fit_lda(self, X : array.array, y : array.array) -> None:
        """
        Optional second pass, after finalize. Add a batch to the LDA statistics.
        Call it for every batch of the training data, then call finalize_lda.

        :param X: Batch features, float32 array of n_samples * n_features
        :param y: Batch targets as class indices, float32 array of n_samples
        """
        pass

    def finalize_lda(self) -> None:
        """
        Fit LDA on the class scores and update the weights and biases.
        """
        pass

    def predict(self, features : array.array, scores : array.array) -> int:
        """
        Run prediction and return the predicted class index.

        :param features: Input features, float32 array of n_features
        :param scores: Output buffer for class scores, float32 array of n_classes (modified in-place)
        :return: Index of the predicted class
        """
        pass

    def reset(self) -> None:
        """
        Forget all data seen so far.
        """
        pass

    def get_weights(self, out : array.array) -> None:
        """
        Copy the model weights into an output buffer.

        :param out: Float32 buffer of n_features * n_classes (modified in-place)
        """
        pass

    def get_bias(self, out : array.array) -> None:
        """
        Copy the model biases into an output buffer.

        :param out: Float32 buffer of n_classes (modified in-place)
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

    def __del__(self) -> None:
        pass


def new(n_features : int, n_classes : int) -> Model:
    """
    Construct a new SEFR model.

    :param n_features: Number of input features
    :param n_classes: Number of output classes, at least 2
    """
    pass


def train(model : Model, X : array.array, y : array.array, lda : bool = ...) -> None:
    """
    Fit on a full dataset, discarding any data seen before.

    :param model: SEFR model instance
    :param X: Training features, float32 array of n_samples * n_features
    :param y: Training targets as class indices, float32 array of n_samples
    :param lda: Also run the LDA pass over the same data
    """
    pass
