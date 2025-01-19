
# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
Infinite Impulse Response (IIR) filters 

Uses a cascade of second-order filter sections (SOS).
The conventions are designed to match that of scipy-signal (https://docs.scipy.org/doc/scipy/reference/signal.html),
so one can use design tools such as scipy.signal.iirfilter or scipy.signal.iirdesign
to create the IIR filter coefficients.

Implemented using *eml_iir* from the emlearn C library (https://github.com/emlearn/emlearn).
"""

import array
import typing


class IIR():
    """Infinite Impulse Response filter
    """
    def run(self, values : array.array):
        """
        Perform the filter

        Note: operates in-place, will modify the input array data.

        :param values: the data to filter. Typecode 'f' (float)
        """
        pass


def new(coefficients : array.array) -> IIR:
    """
    Create IIR filter

    There must be 6 coefficients per second-order stage.
    The format of each stage is on form Transposed Direct Form II: (b0, b1, b2, 1.0, -a1, -a2).
    Multiple stages are formed by concatenating the coefficients of each stage.
    This is the same used by scipy.signal.sosfilt.

    :param coefficients: IIR filter coefficients
    """
    pass


