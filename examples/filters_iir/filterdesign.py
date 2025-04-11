
"""
IIR filter design
"""

import math
import cmath

PI = math.pi

def prewarp(f, fs):
    return 2 * fs * math.tan(PI * f / fs)

def butterworth_poles(n):
    poles = []
    for k in range(1, n + 1):
        angle = PI * (2 * k + n - 1) / (2 * n)
        s = -math.sin(angle) + 1j * math.cos(angle)
        poles.append(s)
    return poles

def bilinear(poles, zeros, fs):
    T = 1 / fs
    def bilinear_map(s):
        return (2 + s * T) / (2 - s * T)
    poles_z = [bilinear_map(p) for p in poles]
    zeros_z = [bilinear_map(z) if z != float('inf') else -1 for z in zeros]
    return poles_z, zeros_z

def sos_from_poles_zeros(poles, zeros, gain=1.0):
    sos = []
    poles = list(poles)
    zeros = list(zeros)

    # Pad with extra zeros at z = -1 if needed
    while len(zeros) < len(poles):
        zeros.append(-1)

    poles_pairs = [poles[i:i+2] for i in range(0, len(poles), 2)]
    zeros_pairs = [zeros[i:i+2] for i in range(0, len(zeros), 2)]

    for zp, pp in zip(zeros_pairs, poles_pairs):
        b = poly(zp)
        a = poly(pp)
        sos.append([b[0]*gain, b[1]*gain, b[2]*gain, a[0], a[1], a[2]])
        gain = 1.0  # Only apply gain to first section
    return sos

def poly(roots):
    # Return polynomial coefficients from roots
    if len(roots) == 1:
        r = roots[0]
        return [1.0, -r.real, 0.0]
    elif len(roots) == 2:
        r1, r2 = roots
        b0 = 1.0
        b1 = -(r1 + r2).real
        b2 = (r1 * r2).real
        return [b0, b1, b2]
    else:
        return [1.0, 0.0, 0.0]

def design_butterworth(order, ftype, fs, f1, f2=None, gain = 1.0):
    analog_poles = butterworth_poles(order)

    if ftype == 'lowpass':
        warped = prewarp(f1, fs)
        poles = [p * warped for p in analog_poles]
        zeros = [float('inf')] * order

    elif ftype == 'highpass':
        warped = prewarp(f1, fs)
        poles = [warped / p for p in analog_poles]
        zeros = [0.0] * order

    elif ftype == 'bandpass':
        warped1 = prewarp(f1, fs)
        warped2 = prewarp(f2, fs)
        bw = warped2 - warped1
        w0 = math.sqrt(warped1 * warped2)
        poles_bp = []
        for p in analog_poles:
            s1 = 0.5 * bw * p + cmath.sqrt((0.5 * bw * p)**2 - w0**2)
            s2 = 0.5 * bw * p - cmath.sqrt((0.5 * bw * p)**2 - w0**2)
            poles_bp.extend([s1, s2])
        poles = poles_bp
        zeros = [0.0] * (order * 2)

    else:
        raise ValueError("Unsupported filter type: choose 'lowpass', 'highpass', or 'bandpass'")

    # Bilinear transform to digital
    poles_z, zeros_z = bilinear(poles, zeros, fs)

    # Return second-order sections
    return sos_from_poles_zeros(poles_z, zeros_z, gain)
