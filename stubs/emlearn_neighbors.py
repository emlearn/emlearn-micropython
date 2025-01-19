
# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
K-nearest neighbors

Implemented using *eml_neighbors* from the emlearn C library (https://github.com/emlearn/emlearn).
"""

import array
import typing


class Model():
    """A nearest-neighbors model
    """
    def predict(self, inputs : array.array) -> int:
        """
        Run inference using the model

        :param inputs: the input data. Typecode 'h' (int16)
        :return: the resulting label/class
        """
        pass

    def additem(self, values : array.array, label : int):
        """
        Add an item into the model

        :param values: the data/features of this item. Typecode 'h' (int16)
        :param label: the label/class to associate with this item
        """
        pass

    def getitem(self, item : int, outputs : array.array):
        """
        Access data of an item stored in the model

        :param item: Index of item
        :param outputs: Where to copy the data from the item. Typecode 'h' (int16)
        """
        pass

    def getresult(self, idx : int) -> tuple[int, int, int]:
        """
        Get details on the comparisons between predict() data and items stored in model

        :param item: Index of the comparison to retrieve. Smaller number are the nearest neighbors.
        :return: Tuple with (item-index, distance-to-item, label-of-item)
        """
        pass

def new(max_items : int, features : int, k_neighbors : int) -> Model:
    """
    Construct an empty neighbors model

    The model is created with a specified maximum capacity.
    Memory usage will be determined by this capacity.

    :param max_items: Maximum number of items in the dataset
    :param features: Number of features in a data item
    :param k_neighbors: Number of neighbors to consider
    """
    pass


