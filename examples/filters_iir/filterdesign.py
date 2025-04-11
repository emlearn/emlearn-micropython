
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
    omega_c = 2 * math.pi * fc
    omega_d = 2 * fs * math.tan(omega_c / (2 * fs))
    
    # Scale poles and zeros by prewarped frequency
    scaled_poles = [p * omega_d for p in s_poles]
    scaled_zeros = [z * omega_d for z in s_zeros]
    
    # Apply bilinear transform
    z_poles = [(1 + p/(2*fs)) / (1 - p/(2*fs)) for p in scaled_poles]
    z_zeros = [(1 + z/(2*fs)) / (1 - z/(2*fs)) for z in scaled_zeros]
    
    return z_poles, z_zeros

def normalize_gain(z_poles, z_zeros, filter_type, fc, fs):
    """
    Calculate the gain to normalize the filter response at DC or Nyquist.
    
    Args:
        z_poles: Z-plane poles
        z_zeros: Z-plane zeros
        filter_type: 'lowpass', 'highpass', or 'bandpass'
        fc: Cutoff frequency
        fs: Sampling frequency
        
    Returns:
        Gain factor
    """
    # Evaluate the gain at the appropriate frequency
    if filter_type == 'lowpass':
        z = 1.0  # DC (z = 1)
    elif filter_type == 'highpass':
        z = -1.0  # Nyquist (z = -1)
    else:  # bandpass
        # For bandpass, we'll evaluate at the center frequency
        z = complex(math.cos(2 * math.pi * fc[0] / fs), math.sin(2 * math.pi * fc[0] / fs))
    
    # Calculate gain
    numerator = 1.0
    for zero in z_zeros:
        numerator *= abs(z - zero)
    
    denominator = 1.0
    for pole in z_poles:
        denominator *= abs(z - pole)
    
    if numerator == 0:
        return 1.0
    
    return denominator / numerator

def group_into_sos(poles, zeros, gain):
    """
    Group poles and zeros into second-order sections.
    
    Args:
        poles: List of poles
        zeros: List of zeros
        gain: Overall gain
        
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
    
    # Apply gain to the first section
    section_gain = gain ** (1 / ((n_poles + 1) // 2))
    
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
                b0 = section_gain
                b1 = -2 * z1.real * b0  # Sum of conjugate pair
                b2 = (abs(z1) ** 2) * b0  # Product of conjugate pair
            else:
                # Real zeros
                b0 = section_gain
                b1 = -(z1 + z2) * b0
                b2 = (z1 * z2) * b0
            
            sos_list.append([b0, b1, b2, a0, a1, a2])
            i += 2
            
            # Reset gain for the remaining sections
            section_gain = 1.0
        else:
            # Real pole
            p = poles[i]
            z = zeros[i]
            
            # First-order section
            a0 = 1.0
            a1 = -p
            a2 = 0.0
            
            b0 = section_gain
            b1 = -z * b0
            b2 = 0.0
            
            sos_list.append([b0, b1, b2, a0, a1, a2])
            i += 1
            
            # Reset gain for the remaining sections
            section_gain = 1.0
    
    return sos_list

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
    
    # For lowpass filters, zeros that were at infinity in s-plane map to z=-1
    z_zeros = [-1.0] * order
    
    # Calculate gain for normalization
    gain = normalize_gain(z_poles, z_zeros, 'lowpass', cutoff, fs)
    
    # Group into second-order sections
    sos = group_into_sos(z_poles, z_zeros, gain)
    
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
    
    # For highpass filters, zeros at s=0 map to z=1
    z_zeros = [1.0] * order
    
    # Calculate gain for normalization
    gain = normalize_gain(z_poles, z_zeros, 'highpass', cutoff, fs)
    
    # Group into second-order sections
    sos = group_into_sos(z_poles, z_zeros, gain)
    
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
    z_poles, z_zeros = s_to_z_bilinear(bp_s_poles, bp_s_zeros, fs, (cutoff[0] + cutoff[1])/2)
    
    # For bandpass filters, zeros at s=0 map to z=1 and zeros at s=inf map to z=-1
    z_zeros = [1.0] * order + [-1.0] * order
    
    # Calculate gain for normalization
    gain = normalize_gain(z_poles, z_zeros, 'bandpass', cutoff, fs)
    
    # Group into second-order sections
    sos = group_into_sos(z_poles, z_zeros, gain)
    
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

