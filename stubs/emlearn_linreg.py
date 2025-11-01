# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
Linear Regression with support for training/learning/fitting as well as inference/predictions.

Supports combined L1 and L2 regularization, often called ElasticNet.
"""

import array
import typing

class Model():
    """A linear-regression model

    Note: Use emlearn_linreg.new to construct an instance
    """
    def predict(self, inputs : array.array) -> float:
        """
        Run inference using the model

        :param inputs: the input data. Typecode 'f' (float)
        :return: the predicted value
        """
        pass

    def get_n_features(self) -> int:
        """
        Get the number of features the model expects

        :return: Number of features
        """
        pass


    def get_weights(self, weights : array.array):
        """
        Access data of an item stored in the model

        :param weights: Where to copy the weights. Must be n_features long.
        """
        pass

    def set_weights(self, weights : array.array):
        """
        Access data of an item stored in the model

        :param weights: The weights to use. Must be n_features long.
        """
        pass

    def get_bias(self) -> float:
        """
        Get the bias/intercept
        """
        pass

    def set_bias(self, bias : float):
        """
        Set the bias/intercept
        """
        pass


    def step(self, X, y) -> None:
        """
        Perform a single gradient decent step for training/fitting model.

        :param X: Features for regression. Must be n_features*n_rows long
        :param y: Targets for regression. Must be n_rows long
        """
        pass

    def score_mse(self, X) -> float:
        """
        Compute Mean Squared Error (MSE) on a set of samples.

        :param X: Features for regression. Must be n_features*n_rows long
        :return: The MSE score
        """
        pass

def new(features : int, alpha : float,  l1_ratio: float, learning_rate : float) -> Model:
    """
    Construct an new linear regression model

    :param features: Number of features in a data item
    :param k_neighbors: Number of neighbors to consider
    :param l1_ratio: Balance between L2 and L1 loss
    :param learning_rate: Learning rate to use during optimization
    """
    pass

def train(model,
        X_train : array.array,
        y_train : array.array,
        max_iterations=100, 
        tolerance=1e-6,
        check_interval=10,
        divergence_factor=10.0,
        score_limit=None,
        verbose=0):
    """
    Simple training loop using Mean Squared Error

    Runs gradient decent iteratively until a tolerance has been achieved, a score reached, or max_iterations.
    For more complicated training needs, copy this code as an example starting point.

    :param model: emlearn_linreg instance to train
    :param X_train: Features for regression. Must be n_features*n_rows long
    :param y_train: Targets for regression. Must be n_rows long
    :param model: emlearn_linreg instance to train
    :param max_iterations: Maximum number of training steps
    :param tolerance: If change in score between checks is below this limit, consider converged.
    :param check_interval: How many steps between each check of convergence/divergence
    :param score_limit: If score is beow this limit, consider converged.
    :param verbose: Whether to print/log outputs
    """

