# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
Convolutional Neural Network module.

Implemented using TinyMaix (https://github.com/sipeed/TinyMaix/).

Note that multiple variants of this module is provided.
So the import should either be **emlearn_cnn_fp32** (for 32 bit floating point),
or **emlearn_cnn_int8** (for 8 bit integers, quantized model).

"""

import array

class Model():
    """A Convolutional Neural Network

    Note: May not be constructed directly. Instead use the new() function.
    """
    def run(self, inputs : array.array, outputs: array.array):
        """
        Run inference using the model

        :param inputs: the input data
        :param outputs: where to put model outputs
        """
        pass

    def output_dimensions(self) -> tuple[int]:
        """
        Get the output dimensions/size of the model

        Useful to know how large an array to pass to run()
        """
        pass

def new(model_data : array.array) -> Model:
    """
    Construct a new Convolutional Neural Network

    :param model_data: The model definition (.TMDL file, as bytes).
    """
    pass

