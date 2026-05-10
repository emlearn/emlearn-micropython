
# Stub file (PEP 484) with API definitions and documentation for native module
# Is called .py because Sphinx autodoc currently does not support .pyi files

"""
Fast Fourier Transform (FFT)

Implemented using *eml_fft* from the emlearn C library (https://github.com/emlearn/emlearn).
"""

import array
import typing


class FFT():
    """Fast Fourier Transform (FFT)
    """
    def __init__(self, length : int):
        pass

    def run(self, real : array.array, imag : array.array):
        """
        Perform the FFT transformation

        Note: operates in-place, will modify both arrays.

        To perform a real-only (RFFT), let imag be all zeroes.

        :param real: the real part of data. Typecode 'f' (float)
        :param imag: the imaginary part of data. Typecode 'f' (float)
        """
        pass

    def fill(self, sin : array.array, cos : array.array):
        """
        Set up FFT coefficients

        Note: Do not use this directly. Instead use the module-level fill() function.

        :param sin: Precomputed coefficients
        :param cos: Precomputed coefficients 
        """
        pass

def fill(fft : FFT, n : int):
    """
    Set up FFT coefficients

    NB: Must be called before attempting to use FFT.run()

    :param fft: FFT instance
    :param n: Length of the FFT transform
    """
    pass


