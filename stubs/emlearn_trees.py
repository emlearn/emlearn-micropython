# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
Tree-based models (Random Forest et.c.)

Implemented using emlearn C library (https://github.com/emlearn/emlearn).
"""

import array
import typing

class Model():
    """A tree-based ensemble model

    Note: Normally not constructed directly. Instead use
    """
    def predict(self, inputs : array.array, outputs: array.array):
        """
        Run inference using the model

        :param inputs: the input data. Typecode 'h' (int16)
        :param outputs: where to put model outputs. Typecode 'f' (float)
        """
        pass

    def outputs(self) -> int:
        """
        Get the output dimensions/size of the model

        Useful to know how large an array to pass to predict()
        """
        pass

    def setdata(self, features : int, classes : int):
        """
        Set data about the model

        Note: Usually not used directly. Instead use load_model().

        :param features: Number of input features
        :param classes: Number of classes
        """
        pass

    def addroot(self, root):
        """
        Add a tree root

        Note: Usually not used directly. Instead use load_model().

        :param root: Offset into nodes for the initial decision node of a tree
        """
        pass

    def addnode(self, left : int, right : int, feature : int, value : int):
        """
        Add a decision node

        Note: Usually not used directly. Instead use load_model().

        :param left: Left child (node or leaf)
        :param right: Right child (node or leaf)
        :param feature: Feature index 
        :param value: Threshold to compute feature to
        """
        pass

    def addleaf(self, value : int):
        """
        Add a leaf node

        Note: Usually not used directly. Instead use load_model().

        :param value: 
        """
        pass

def new(max_trees : int, max_nodes : int, max_leaves : int) -> Model:
    """
    Construct an empty tree-based model

    The model is created with a specified maximum capacity.
    Memory usage will be determined by this capacity.

    :param max_trees: Maximum number of trees in ensemble
    :param max_nodes: Maximum number of decision nodes (across all trees)
    :param max_leaves: Maximum number of leaves (across all trees)
    """
    pass

def load_model(trees : Model, file : typing.BinaryIO):

    """
    Load model definition from a file

    The model must be constructed with sufficient capacity (trees, nodes, leaves).
    Otherwise will raise exception.
    """
    pass

