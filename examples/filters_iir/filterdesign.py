import math
import cmath

def butter_poles(n):
    """
    Calculate the poles of an nth-order Butterworth filter in the s-plane.
    
    Args:
        n: Filter order
        
    Returns:
        List of complex poles in the s-plane
    """
    poles = []
    for k in range(n):
        theta = math.pi * (2*k + n + 1) / (2*n)
        pole = complex(math.cos(theta), math.sin(theta))
        poles.append(pole)
    return poles

def s_to_z_bilinear(s_poles, s_zeros, fs, fc):
    """
    Convert s-plane poles and zeros to z-plane using bilinear transform.
    
    Args:
        s_poles: List of s-plane poles
        s_zeros: List of s-plane zeros
        fs: Sampling frequency
        fc: Cutoff frequency
        
    Returns:
        Tuple of (z_poles, z_zeros)
    """
    # Prewarp cutoff frequency
    if isinstance(fc, tuple):
        # For bandpass, use geometric mean of cutoffs
        omega_c = 2 * math.pi * math.sqrt(fc[0] * fc[1])
    else:
        omega_c = 2 * math.pi * fc
    
    omega_d = 2 * fs * math.tan(omega_c / (2 * fs))
    
    # Scale poles and zeros by prewarped frequency
    scaled_poles = [p * omega_d for p in s_poles]
    scaled_zeros = [z * omega_d if z != float('inf') else z for z in s_zeros]
    
    # Apply bilinear transform
    z_poles = [(1 + p/(2*fs)) / (1 - p/(2*fs)) for p in scaled_poles]
    
    z_zeros = []
    for z in scaled_zeros:
        if z == float('inf'):
            z_zeros.append(-1.0)  # Infinity maps to -1
        elif z == 0:
            z_zeros.append(1.0)   # Zero maps to 1
        else:
            z_zeros.append((1 + z/(2*fs)) / (1 - z/(2*fs)))
    
    return z_poles, z_zeros

def calculate_dc_gain(sos):
    """
    Calculate the DC gain (z=1) or Nyquist gain (z=-1) of SOS filter.
    
    Args:
        sos: List of second-order sections
        
    Returns:
        DC gain value
    """
    gain = 1.0
    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        # Evaluate at z=1 (DC)
        numerator = b0 + b1 + b2
        denominator = a0 + a1 + a2
        gain *= numerator / denominator
    return gain

def calculate_nyquist_gain(sos):
    """
    Calculate the Nyquist gain (z=-1) of SOS filter.
    
    Args:
        sos: List of second-order sections
        
    Returns:
        Nyquist gain value
    """
    gain = 1.0
    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        # Evaluate at z=-1 (Nyquist)
        numerator = b0 - b1 + b2
        denominator = a0 - a1 + a2
        gain *= numerator / denominator
    return gain

def group_into_sos(poles, zeros):
    """
    Group poles and zeros into second-order sections.
    
    Args:
        poles: List of poles
        zeros: List of zeros
        
    Returns:
        List of SOS coefficients [b0, b1, b2, a0, a1, a2]
    """
    # Ensure we have an even number of poles and zeros (for complex conjugate pairs)
    n_poles = len(poles)
    n_zeros = len(zeros)
    
    # Add zeros at z=-1 if needed to match the number of poles
    while n_zeros < n_poles:
        zeros.append(-1.0)
        n_zeros += 1
    
    # Sort poles by magnitude (ascending)
    poles = sorted(poles, key=lambda x: abs(x))
    zeros = sorted(zeros, key=lambda x: abs(x))
    
    # Group into second-order sections
    sos_list = []
    i = 0
    
    while i < n_poles:
        if i + 1 < n_poles and isinstance(poles[i], complex) and poles[i].imag != 0:
            # Complex conjugate pair
            p1 = poles[i]
            p2 = poles[i+1]
            
            z1 = zeros[i]
            z2 = zeros[i+1]
            
            # Second-order section denominator coefficients
            a0 = 1.0
            a1 = -2 * p1.real  # Sum of conjugate pair
            a2 = abs(p1) ** 2  # Product of conjugate pair
            
            # Second-order section numerator coefficients
            if isinstance(z1, complex) and z1.imag != 0:
                # Complex conjugate zeros
                b0 = 1.0
                b1 = -2 * z1.real  # Sum of conjugate pair
                b2 = abs(z1) ** 2  # Product of conjugate pair
            else:
                # Real zeros
                b0 = 1.0
                b1 = -(z1 + z2)
                b2 = z1 * z2
            
            sos_list.append([b0, b1, b2, a0, a1, a2])
            i += 2
        else:
            # Real pole
            p = poles[i]
            z = zeros[i]
            
            # First-order section
            a0 = 1.0
            a1 = -p
            a2 = 0.0
            
            b0 = 1.0
            b1 = -z
            b2 = 0.0
            
            sos_list.append([b0, b1, b2, a0, a1, a2])
            i += 1
    
    return sos_list

def normalize_sos(sos, filter_type):
    """
    Normalize SOS filter to have unit gain at the appropriate frequency.
    
    Args:
        sos: List of second-order sections
        filter_type: 'lowpass', 'highpass', or 'bandpass'
        
    Returns:
        Normalized SOS coefficients
    """
    if filter_type == 'lowpass':
        # Normalize to have unity gain at DC (z=1)
        dc_gain = calculate_dc_gain(sos)
        gain_factor = 1.0 / dc_gain
    elif filter_type == 'highpass':
        # Normalize to have unity gain at Nyquist (z=-1)
        nyquist_gain = calculate_nyquist_gain(sos)
        gain_factor = 1.0 / nyquist_gain
    else:  # bandpass
        # For bandpass, we'll normalize the gain at DC to be 0,
        # and the peak gain to be 1.0
        # This is done in the butter_bandpass function itself
        return sos
    
    # Apply gain normalization to the first section
    normalized_sos = sos.copy()
    b0, b1, b2, a0, a1, a2 = normalized_sos[0]
    normalized_sos[0] = [b0 * gain_factor, b1 * gain_factor, b2 * gain_factor, a0, a1, a2]
    
    return normalized_sos

def butter_lowpass(order, cutoff, fs):
    """
    Design a digital Butterworth lowpass filter.
    
    Args:
        order: Filter order
        cutoff: Cutoff frequency
        fs: Sampling frequency
        
    Returns:
        SOS coefficients
    """
    # Get analog filter poles
    s_poles = butter_poles(order)
    
    # Lowpass filter has zeros at infinity in s-plane (z=-1 in z-plane)
    s_zeros = [float('inf')] * order
    
    # Convert to digital filter via bilinear transform
    z_poles, z_zeros = s_to_z_bilinear(s_poles, s_zeros, fs, cutoff)
    
    # Group into second-order sections
    sos = group_into_sos(z_poles, z_zeros)
    
    # Normalize to have unity gain at DC
    sos = normalize_sos(sos, 'lowpass')
    
    return sos

def butter_highpass(order, cutoff, fs):
    """
    Design a digital Butterworth highpass filter.
    
    Args:
        order: Filter order
        cutoff: Cutoff frequency
        fs: Sampling frequency
        
    Returns:
        SOS coefficients
    """
    # Get analog filter poles
    s_poles = butter_poles(order)
    
    # Map lowpass to highpass in the s-plane: s -> 1/s
    s_poles = [1.0/p for p in s_poles]
    
    # Highpass filter has zeros at s=0 (z=1 in z-plane)
    s_zeros = [0.0] * order
    
    # Convert to digital filter via bilinear transform
    z_poles, z_zeros = s_to_z_bilinear(s_poles, s_zeros, fs, cutoff)
    
    # Group into second-order sections
    sos = group_into_sos(z_poles, z_zeros)
    
    # Normalize to have unity gain at Nyquist
    sos = normalize_sos(sos, 'highpass')
    
    return sos

def butter_bandpass(order, cutoff, fs):
    """
    Design a digital Butterworth bandpass filter.
    
    Args:
        order: Filter order (will be doubled internally)
        cutoff: Tuple of (low_cutoff, high_cutoff)
        fs: Sampling frequency
        
    Returns:
        SOS coefficients
    """
    # For bandpass filter, the order will be doubled
    bp_order = 2 * order
    
    # Get analog filter poles
    s_poles = butter_poles(order)
    
    # Calculate center frequency and bandwidth in radians/sec
    w0 = 2 * math.pi * math.sqrt(cutoff[0] * cutoff[1])
    bw = 2 * math.pi * (cutoff[1] - cutoff[0])
    
    # Transform poles for bandpass filter
    bp_s_poles = []
    for p in s_poles:
        # Each pole generates two new poles
        bp_s_poles.append(0.5 * bw * p + cmath.sqrt(0.25 * (bw * p)**2 - w0**2))
        bp_s_poles.append(0.5 * bw * p - cmath.sqrt(0.25 * (bw * p)**2 - w0**2))
    
    # Bandpass filter has zeros at s=0 and s=infinity
    bp_s_zeros = [0.0] * order + [float('inf')] * order
    
    # Convert to digital filter via bilinear transform
    z_poles, z_zeros = s_to_z_bilinear(bp_s_poles, bp_s_zeros, fs, cutoff)
    
    # Group into second-order sections
    sos = group_into_sos(z_poles, z_zeros)
    
    # For bandpass, we need to handle normalization differently
    # Let's find the frequency response at the center frequency
    center_freq = math.sqrt(cutoff[0] * cutoff[1])
    z_center = cmath.rect(1.0, 2 * math.pi * center_freq / fs)
    
    # Calculate gain at center frequency
    gain = 1.0
    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        numerator = b0 + b1 * z_center**-1 + b2 * z_center**-2
        denominator = a0 + a1 * z_center**-1 + a2 * z_center**-2
        gain *= abs(numerator / denominator)
    
    # Apply gain normalization to the first section
    gain_factor = 1.0 / gain
    b0, b1, b2, a0, a1, a2 = sos[0]
    sos[0] = [b0 * gain_factor, b1 * gain_factor, b2 * gain_factor, a0, a1, a2]
    
    return sos

def sosfreqz(sos, worN=512, fs=2*math.pi):
    """
    Compute the frequency response of a digital filter in SOS format.
    
    Args:
        sos: List of second-order sections coefficients
        worN: Number of frequency points
        fs: Sampling frequency
        
    Returns:
        Tuple of (frequencies, complex frequency response)
    """
    w = [(fs * i) / (2.0 * worN) for i in range(worN)]
    h = [complex(1.0, 0.0) for _ in range(worN)]
    
    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        for i in range(worN):
            # Compute z = e^(j*omega)
            z = cmath.rect(1.0, math.pi * w[i] / (fs/2))
            
            # Compute frequency response for this section
            numerator = b0 + b1 * z**-1 + b2 * z**-2
            denominator = a0 + a1 * z**-1 + a2 * z**-2
            h[i] *= numerator / denominator
    
    return w, h


def design_butterworth(order, ftype, fs, f1, f2=None, gain=1.0):

    if ftype == 'lowpass':
        sos = butter_lowpass(order, cutoff=f1, fs=fs)
    elif ftype == 'highpass':
        sos = butter_highpass(order, cutoff=f1, fs=fs)
    elif ftype == 'bandpass':
        sos = butter_bandpass(order, cutoff=(f1, f2), fs=fs)
    else:
        raise ValueError("Unsupported filter type")

    return sos

