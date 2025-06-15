
import array
import math

DATA_TYPECODE = 'h' # int16


def feature_names() -> list[str]:
    cls = Feature

    features = {} # index -> name
    for key, value in cls.__dict__.items():
        if key.startswith('__') or callable(value):
            continue
        if features.get(value):
            raise ValueError(f"Duplicated value {value}")
        features[value] = key

    names = []
    for idx, value in enumerate(sorted(features.keys())):
        if idx != value:
            raise ValueError(f'Holes in enum')

        names.append(features[value])

    return names

class Feature:
    orient_w = 0
    orient_x = 1
    orient_y = 2
    orient_z = 3
    sma = 4
    mag_rms = 5
    x_rms = 6
    y_rms = 7
    z_rms = 8

N_FEATURES = 9


def compute_sma(x_data : array.array, y_data : array.array, z_data : array.array) -> float:
    """Signal Magnitude Area (SMA)"""

    n = len(x_data)
    sma_sum = 0.0
    for x, y, z in zip(x_data, y_data, z_data):
        sma_sum += abs(x) + abs(y) + abs(z)

    sma = sma_sum / n
    return sma

def compute_magnitude_rms(x_data : array.array, y_data : array.array, z_data : array.array) -> float:

    n = len(x_data)
    agg = 0.0
    for x, y, z in zip(x_data, y_data, z_data):
        agg += (x*x) + (y*y) + (z*z)

    rms = math.sqrt(agg / n)
    return rms

def compute_rms(data : array.array) -> tuple[float]:

    n = len(data)
    agg = 0.0
    for v in data:
        agg += (v*v)

    rms = math.sqrt(agg / n)
    return rms

def compute_pitch_roll(x : float, y : float, z : float) -> tuple[float]:
    """In degrees"""

    roll = round(math.degrees(math.atan2(y, z)), 2)
    denominator = math.sqrt(y * y + z * z)
    pitch = round(math.degrees(math.atan2(-x, denominator)), 2)

    return pitch, roll


def calculate_features_xyz(xyz : tuple[array.array]) -> array.array:

    xo, yo, zo = xyz

    if not (len(xo) == len(yo) == len(zo)):
        raise ValueError("Input data lists must have the same length.")

    window_length = len(xo)

    # Output array
    feature_data = array.array('f', (0 for i in range(N_FEATURES)))


    # Gravity separation
    # FIXME: replace mean with a low-pass filter at 0.5 Hz. Say Chebychev 2 order
    # and to subtract the continues values.
    # !! need to be able to initialize IIR filter state to first sample
    gravity_x = sum(xo) / window_length
    gravity_y = sum(yo) / window_length
    gravity_z = sum(zo) / window_length

    linear_x = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    linear_y = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    linear_z = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))


    for i in range(window_length):
        i = int(i) # XXX ??
        linear_x[i] = int(xo[i] - gravity_x)
        linear_y[i] = int(yo[i] - gravity_y)
        linear_z[i] = int(zo[i] - gravity_z)

    # Orientation
    # normalize gravity vector
    #gravity_magnitude = math.sqrt(gravity_x*gravity_x + gravity_y*gravity_y + gravity_z*gravity_z)
    #ox = gravity_x/gravity_magnitude
    #oy = gravity_y/gravity_magnitude
    #oz = gravity_z/gravity_magnitude
    #print(ox, oy, oz)

    #roll, pitch = compute_pitch_roll(ox, oy, oz)
    #feature_data[Feature.pitch] = pitch
    #feature_data[Feature.roll] = roll

    orientation_q = tilt_quaternion_from_accel(gravity_x, gravity_y, gravity_z)
    feature_data[Feature.orient_w] = orientation_q[0]
    feature_data[Feature.orient_x] = orientation_q[1]
    feature_data[Feature.orient_y] = orientation_q[2]
    feature_data[Feature.orient_z] = orientation_q[3]


    # Overall motion
    #feature_data[Feature.sma] = compute_sma(linear_x, linear_y, linear_z)
    #feature_data[Feature.mag_rms] = compute_magnitude_rms(linear_x, linear_y, linear_z)

    # Per-axis motion
    #feature_data[Feature.x_rms] = compute_rms(linear_x)
    #feature_data[Feature.y_rms] = compute_rms(linear_y)
    #feature_data[Feature.z_rms] = compute_rms(linear_z)

    print(orientation_q)


    return feature_data


def normalize(v):
    mag = math.sqrt(sum(c*c for c in v))
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return tuple(c / mag for c in v)

def dot(v1, v2):
    return sum(a*b for a, b in zip(v1, v2))

def cross(v1, v2):
    return (
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    )

def tilt_quaternion_from_accel(a_x, a_y, a_z):
    # Gravity vector measured by accelerometer (assumed already low-pass filtered)
    g = normalize((a_x, a_y, a_z))

    # Reference "down" vector in world space
    ref = (0.0, 0.0, 1.0)

    # Compute axis and angle between vectors
    axis = cross(ref, g)
    axis_mag = math.sqrt(sum(c*c for c in axis))

    if axis_mag < 1e-6:
        # Vectors are nearly aligned or opposite
        if dot(ref, g) > 0.999:
            # Identity rotation (device is flat, facing up)
            return (1.0, 0.0, 0.0, 0.0)
        else:
            # 180° rotation around X or Y (choose arbitrary orthogonal axis)
            return (0.0, 1.0, 0.0, 0.0)  # Flip around X

    axis = normalize(axis)
    angle = math.acos(max(-1.0, min(1.0, dot(ref, g))))  # Clamp dot product to avoid domain error

    # Convert axis-angle to quaternion
    half_angle = angle / 2
    sin_half = math.sin(half_angle)
    q_w = math.cos(half_angle)
    q_x = axis[0] * sin_half
    q_y = axis[1] * sin_half
    q_z = axis[2] * sin_half

    return (q_w, q_x, q_y, q_z)


def test_compute():

    window_length = 50

    x = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    y = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    z = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))

    features = calculate_features_xyz((x, y, z))

    names = feature_names()
    assert len(names) == len(features)


def test_pitch_rotation():

    print("\n--- Simulating Pitch (X-axis) Rotation ---")
    # Rotate around X-axis (pitch)
    for angle_deg in range(-90, 91, 15):
        rad = math.radians(angle_deg)
        # Simulated gravity vector for X-axis rotation
        a_x = math.sin(rad)
        a_y = 0
        a_z = math.cos(rad)

        a_xn, a_yn, a_zn = normalize(a_x, a_y, a_z)
        pitch, roll  = compute_pitch_roll(a_xn, a_yn, a_zn)

        print(f"{angle_deg:6} | {a_xn:+.2f} {a_yn:+.2f} {a_zn:+.2f} | {roll:+6.1f}° {pitch:+6.1f}°")

def test_roll_rotation():

    print("\n--- Simulating Roll (Y-axis) Rotation ---")
    for angle_deg in range(-90, 91, 15):
        rad = math.radians(angle_deg)
        a_x = 0
        a_y = math.sin(rad)
        a_z = math.cos(rad)

        a_xn, a_yn, a_zn = normalize(a_x, a_y, a_z)
        pitch, roll  = compute_pitch_roll(a_xn, a_yn, a_zn)

        print(f"{angle_deg:6} | {a_xn:+.2f} {a_yn:+.2f} {a_zn:+.2f} | {roll:+6.1f}° {pitch:+6.1f}°")


if __name__ == '__main__':

    #test_compute()
    test_pitch_rotation()
    test_roll_rotation()

    print('PASS')
