import math
import cmath

PI = math.pi

def prewarp(f, fs):
    return 2 * fs * math.tan(PI * f / fs)

def butterworth_poles(n):
    poles = []
    for k in range(1, n + 1):
        angle = PI * (2 * k - 1) / (2 * n)
        s = -cmath.exp(1j * angle)  # Correct pole location
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
    """Computes polynomial coefficients from given roots."""
    coeffs = [1.0]
    for r in roots:
        new_coeffs = [0.0] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            new_coeffs[i]     -= coeffs[i] * r
            new_coeffs[i + 1] += coeffs[i]
        coeffs = new_coeffs
    return [c.real if isinstance(c, complex) else c for c in coeffs]

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
        sos.append([b[0], b[1], b[2], a[0], a[1], a[2]])
    return sos

def compute_gain(poles_z, zeros_z, ftype):
    """Normalizes gain at DC (for lowpass) or Nyquist (for highpass)."""
    if ftype == 'lowpass':
        z_eval = 1.0
    elif ftype == 'highpass':
        z_eval = -1.0
    else:
        return 1.0  # For bandpass or general cases, skip normalization

    num = 1.0
    den = 1.0
    for z in zeros_z:
        num *= abs(z_eval - z)
    for p in poles_z:
        den *= abs(z_eval - p)
    return num / den if den != 0 else 1.0

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

    # Normalize gain
    norm_gain = compute_gain(poles_z, zeros_z, ftype)
    final_gain = gain / norm_gain

    return sos_from_poles_zeros(poles_z, zeros_z, final_gain)

