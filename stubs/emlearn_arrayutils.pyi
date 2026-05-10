
# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
Array utility functions 

Some simple operations on arrays, implemented in C for efficiency.
"""

import array
import typing

def linear_map(inputs : array.array, outputs: array.array,
        in_min : float, in_max : float,
        out_min : float, out_max : float
    ):
    """
    Compute an element-wise linear mapping/scaling. 

    The range (in_min, in_max) will be mapped to (out_min, out_max).
    The function supports in and out being different data widths (array typecodes).

    The following pairs are supported:
    - float (f) to int16 (h)
    - int16 (h) to float (f)

    :param inputs: The input data
    :param outputs: Where output is placed
    :param in_min: 
    :param in_max: 
    :param out_min: 
    :param out_max: 
    """
    pass


