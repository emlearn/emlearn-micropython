
import math
import array
import gc


def fill(fft, n):
    # pre-compute the trigonometrics needed for FFT computation
    
    coeffs = n // 2

    # pre-compute the constant part of expression
    PI2_N = (math.pi * 2.0) / n

    # compute coeffeicients
    sin = array.array('f', (math.sin(PI2_N * i) for i in range(coeffs)) )
    cos = array.array('f', (math.cos(PI2_N * i) for i in range(coeffs)) )

    # pass it to the C module
    fft.fill(sin, cos)

    # make sure temporary arrays get freed
    gc.collect()
