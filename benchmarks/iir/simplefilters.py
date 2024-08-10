
"""
Filter design for simple low/high/band-pass filters
"""

import math

# https://stackoverflow.com/a/20932062
def butter2_lowpass(f, sr):
    ff = f / sr
    ita = 1.0/ math.tan(math.pi*ff)
    q = math.sqrt(2.0)
    b0 = 1.0 / (1.0 + q*ita + ita*ita)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita*ita - 1.0) * b0
    a2 = -(1.0 - q*ita + ita*ita) * b0

    # Return in biquad / Second Order Stage format
    # to be compatible with scipy, a1 and a2 needed to be flipped??
    sos = [ b0, b1, b2, 1.0, -a1, -a2 ]

    return sos

# https://stackoverflow.com/a/39240059
def butter2_highpass(f, sr):

    # low-pass filter prototype
    ff = f / sr
    ita = 1.0/ math.tan(math.pi*ff)
    q = math.sqrt(2.0)
    b0 = 1.0 / (1.0 + q*ita + ita*ita)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita*ita - 1.0) * b0
    a2 = -(1.0 - q*ita + ita*ita) * b0

    # convert to high-pass
    b0 = b0*ita*ita;
    b1 = -b1*ita*ita;
    b2 = b2*ita*ita;

    # Return in biquad / Second Order Stage format
    # to be compatible with scipy, a1 and a2 needed to be flipped??
    sos = [ b0, b1, b2, 1.0, -a1, -a2 ]

    return sos

def butter2_bandpass(low, high, fs):

    hp = butter2_highpass(low, fs)
    lp = butter2_lowpass(high, fs)
    bp = hp + lp

    return bp




