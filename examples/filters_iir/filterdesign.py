import math
import cmath

PI = math.pi

def prewarp(f, fs):
    return 2 * fs * math.tan(PI * f / fs)

def butterworth_poles(n):
    poles = []
    for k in range(n):
        theta = PI * (2 * k + 1) / (2 * n)
        s = -cmath.exp(1j * theta)  # unit circle, reflected to LHP
        poles.append(s)
    return poles

def bilinear(poles, zeros, fs):
    T = 1 / fs
    def bilinear_map(s):
        return (2 + s * T) / (2 - s * T)
    poles_z = [bilinear_map(p) for p in poles]
    zeros_z = [bilinear_map(z) if z != float('inf') else -1.0 for z in zeros]
    return poles_z, zeros_z


def poly(roots):
    """Return real-valued coefficients of polynomial from complex conjugate roots."""
    coeffs = [1.0]
    for r in roots:
        coeffs = convolve(coeffs, [1, -r])
    return [c.real for c in coeffs]


def convolve(a, b):
    """Manual convolution of two lists."""
    result = [0.0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]
    return result

def sos_from_poles_zeros(poles, zeros, gain=1.0):
    sos = []
    poles = list(poles)
    zeros = list(zeros)

    # Pad with extra zeros at z = -1 if needed
    while len(zeros) < len(poles):
        zeros.append(-1.0)

    poles_pairs = [poles[i:i+2] for i in range(0, len(poles), 2)]
    zeros_pairs = [zeros[i:i+2] for i in range(0, len(zeros), 2)]

    for i, (zp, pp) in enumerate(zip(zeros_pairs, poles_pairs)):
        b = poly(zp)
        a = poly(pp)
        if i == 0:
            b = [c * gain for c in b]
        # Ensure 3 coefficients
        while len(b) < 3: b.append(0.0)
        while len(a) < 3: a.append(0.0)
        
        # Normalize by a0
        a0 = a[0]
        b = [bi / a0 for bi in b]
        a = [ai / a0 for ai in a]

        sos.append([b[0], b[1], b[2], a[0], a[1], a[2]])
    return sos


def evaluate_transfer_function(b, a, z):
    """Evaluate H(z) = B(z) / A(z) at a specific z (complex)."""
    num = sum(b[i] * (z**(-i)) for i in range(len(b)))
    den = sum(a[i] * (z**(-i)) for i in range(len(a)))
    return num / den if den != 0 else 0.0

def compute_dc_gain(sos):
    """Compute the overall gain of the SOS filter at z=1 (DC)."""
    z = 1.0  # DC
    h = 1.0
    for section in sos:
        b = section[0:3]
        a = section[3:6]
        h *= evaluate_transfer_function(b, a, z)
    return abs(h)

def design_butterworth(order, ftype, fs, f1, f2=None, gain=1.0):
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
        poles = []
        for p in analog_poles:
            term = 0.5 * bw * p
            s1 = term + cmath.sqrt(term**2 - w0**2)
            s2 = term - cmath.sqrt(term**2 - w0**2)
            poles.extend([s1, s2])
        zeros = [0.0] * (2 * order)

    else:
        raise ValueError("Unsupported filter type")

    # Bilinear transform
    poles_z, zeros_z = bilinear(poles, zeros, fs)

    # Initial SOS (gain will be corrected below)
    sos = sos_from_poles_zeros(poles_z, zeros_z, gain=1.0)

    # Normalize DC or Nyquist gain based on filter type
    if ftype == 'lowpass':
        target_z = 1.0
    elif ftype == 'highpass':
        target_z = -1.0
    else:
        target_z = None  # Skip gain normalization for bandpass

    if target_z is not None:
        actual_gain = 1.0
        for section in sos:
            b = section[0:3]
            a = section[3:6]
            actual_gain *= evaluate_transfer_function(b, a, target_z)
        if actual_gain != 0:
            scale = gain / abs(actual_gain)
            # Apply scale to first section only
            sos[0][0] *= scale
            sos[0][1] *= scale
            sos[0][2] *= scale

    return sos

