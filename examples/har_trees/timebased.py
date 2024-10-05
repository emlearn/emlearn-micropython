

# Based on https://github.com/atiselsts/feature-group-selection/blob/10853f32baf731d9dc3654043dc457568e7ecead/datasets/extract-features.py
#
# Modified to separate feaure calculation from IO -- Jon Nordby, 2024
# Modified to run on MicroPython -- Jon Nordby, 2024
#
# Description: extracts features (and transforms from raw acceleration data).
# The input data must alreayd be segmented in windows. For each window, all features are computed.
# Author: Atis Elsts, 2019


import os
import sys
import math
import array

#########################################

L1_NORM = 1
L2_NORM = 2
L2_NORM_SQUARED = 3

#########################################

def scale(v):
    # The data is in range such that 1g == 1.0 in the data.
    # We want to get back to the raw acceleration data (+-4g range),
    # such that 1g == 32 in the data.
    SCALING_FACTOR = 128 // 4
    MAX_VAL = 127
    MIN_VAL = -128

    vi = int(round(float(v) * SCALING_FACTOR, 0))
    if vi > MAX_VAL:
        vi = MAX_VAL
    elif vi < MIN_VAL:
        vi = MIN_VAL
    return vi

def scale_filter(matrix):
    row = matrix
    return [scale(u) for u in row]

#########################################

def normalize(v):
    mn = min(v)
    mx = max(v)
    mean = np.mean(v)
    d = (mx - mn) / 2.0 # from -1 to +1: length 2
    if d:
        return [(x - mean) / d for x in v]
    else:
        return [0.0 for x in v]

#########################################

FEATURE_MEAN = 0
FEATURE_MEDIAN = 1
FEATURE_Q25 = 2
FEATURE_Q75 = 3
FEATURE_IQR = 4
FEATURE_MIN = 5
FEATURE_MAX = 6
FEATURE_STD = 7
FEATURE_ENERGY = 8
FEATURE_ENTROPY = 9
ORDERED_FEATURES_N = 10

def ordered_features(matrix : array.array, results : array.array):
    assert len(results) == ORDERED_FEATURES_N

    WINDOW_SIZE = len(matrix)
    v = matrix
    MEDIAN = WINDOW_SIZE // 2
    Q1 = WINDOW_SIZE // 4
    Q3 = 3 * WINDOW_SIZE // 4

    l = sorted(list(v))
    l2 = [x*x for x in l]
    sm = sum(l)
    sqs = sum(l2)
    avg = sum(l) / len(l)
    median = l[MEDIAN]  
    mn = l[0]
    mx = l[-1]
    energy = ((sqs / len(l2)) ** 0.5) # rms
    std = ((sqs - avg * avg) ** 0.5)

    # FIXME: implement FEATURE_MAD
    #mad_list = [abs(x - l[MEDIAN]) for x in l]
    #mad_list.sort()
    #mad.append(mad_list[MEDIAN])

    # FIXME: implement entropy
    #bins, bin_edges = np.histogram(l, bins=10, density=True)
    #print(scipy.stats.entropy(bins), bins)
    # scipy.stats.entropy(bins)
    entropy = 0 # FIXME: not implemented

    results[FEATURE_MEAN] = avg
    results[FEATURE_MEDIAN] = median
    results[FEATURE_Q25] = l[Q1]
    results[FEATURE_Q75] = l[Q3]
    results[FEATURE_IQR] = (l[Q3] - l[Q1])
    results[FEATURE_MIN] = mn
    results[FEATURE_MAX] = mx
    results[FEATURE_STD] = std
    results[FEATURE_ENERGY] = energy
    results[FEATURE_ENTROPY] = entropy


#########################################

def corr(results, a, b, suffix):
    corrs = []
    for v1, v2 in zip(a, b):
        cc = np.corrcoef(v1, v2)
        r = cc[0][1]
        if math.isnan(r):
            r = 1.0 # std == 0; assume perfect correlation (wise?)
        corrs.append(r)
    results["tTotalAcc-correlation()-" + suffix] = corrs

#########################################w

def jerk_filter(matrix):
    row = matrix

    jrow = [0]
    for i in range(len(row) - 1):
        jrow.append(row[i + 1] - row[i])
    return jrow

#########################################w

def norm_filter(x, y, z, code):
    if code == L1_NORM:
        return [abs(x[i]) + abs(y[i]) + abs(z[i]) for i in range(len(x))]
    elif code == L2_NORM:
        return [(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])**0.5 for i in range(len(x))]
    elif code == L2_NORM_SQUARED:
        return [(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) for i in range(len(x))]
    else:
        raise ValueError('Unsupported norm', code)


##########################################

def median(a, b, c):
    if a > b:
        if b > c:
            return b # a, b, c
        if a > c:
            return c # a, c, b
        return a # c, a, b
    else: # a <= b
        if a > c:
            return a # b, a, c
        if b > c:
            return c # b, c, a
        return b # c, b, a

def median_filter(data):
    row = data

    r = []
    r.append(row[0])
    for i in range(1, len(row) - 1):
        v = median(row[i-1], row[i], row[i+1])
        r.append(v)
    r.append(row[-1])

    return r


def calculate_features_xyz(xyz):

    x, y, z = xyz

    x = median_filter(x)
    y = median_filter(y)
    z = median_filter(z)

    x = scale_filter(x)
    y = scale_filter(y)
    z = scale_filter(z)

    # Doing a derivative is going to reduce the effective recovery data frequency 2 times.
    # This assumes that the data is already low-pass filtered (for 50 Hz to 20 Hz in the dataset)
    # therefore high-frequency components are negligible
    x_jerk = jerk_filter(x)
    y_jerk = jerk_filter(y)
    z_jerk = jerk_filter(z)

    # do the squared L2 norm for now instead of the normal L2 norm
    norm_options = [None, L1_NORM, L2_NORM_SQUARED]
    jerk_options = [False, True]

    l1_norm = norm_filter(x, y, z, L1_NORM)
    l2_norm_sq = norm_filter(x, y, z, L2_NORM_SQUARED)

    l1_norm_jerk = jerk_filter(l1_norm)
    l2_norm_sq_jerk = jerk_filter(l2_norm_sq)

    all_results = []
    all_feature_names = []

    # reusable scratch buffer, to reduce allocations
    features_array = array.array('f', (0 for _ in range(ORDERED_FEATURES_N)))

    for do_jerk in jerk_options:
        if do_jerk:
            tx = x_jerk
            ty = y_jerk
            tz = z_jerk
            tl1 = l1_norm_jerk
            tl2 = l2_norm_sq_jerk
            jerk_name = "Jerk"
        else:
            tx = x
            ty = y
            tz = z
            tl1 = l1_norm
            tl2 = l2_norm_sq
            jerk_name = ""

        results = calculate_features_of_transform(tx, ty, tz, features_array)
        all_results += results

        # jerk_name, "L1Norm"
        results = calculate_features_of_norm_transform(tl1, features_array)
        all_results += results

        # jerk_name, "MagSq"
        results = calculate_features_of_norm_transform(tl2, features_array)

        all_results += results

    assert len(all_results) == 92

    return all_results

# Q25/Q75 dropped
norm_features = [
    FEATURE_MEAN,
    FEATURE_MIN,
    FEATURE_MAX,
    FEATURE_MEDIAN,
    FEATURE_IQR,
    FEATURE_ENERGY,
    FEATURE_STD,
    FEATURE_ENTROPY,
]

def calculate_features_of_norm_transform(m, features_array):
    ordered_features(m, features_array)

    results_list = []
    for n in norm_features:
        results_list.append(features_array[n])

    assert len(results_list) == len(norm_features)

    return results_list


transform_features = [
    FEATURE_MEAN,
    FEATURE_MAX,
    FEATURE_MIN,
    FEATURE_MEDIAN,
    FEATURE_Q25,
    FEATURE_Q75,
    FEATURE_IQR,
    FEATURE_ENERGY,
    FEATURE_STD,
    FEATURE_ENTROPY,
    # TODO: include correlation
    # TODO: include 'all'? type features
    # TODO: inlcude "sma" feature ?
]

def calculate_features_of_transform(x, y, z, features_array : array.array):
    results_list = []

    # X
    ordered_features(x, features_array)
    for n in transform_features:
        results_list.append(features_array[n])

    # Y
    ordered_features(y, features_array)
    for n in transform_features:
        results_list.append(features_array[n])

    # Z
    ordered_features(z, features_array)
    for n in transform_features:
        results_list.append(features_array[n])

    # FIXME: this should be a SUM probably, not a concatenation
    #combined = x + y + z
    #ordered_features(combined, results)

    # FIXME: raises exception
    #corr(results, x, y, "XY")
    #corr(results, x, z, "XZ")
    #corr(results, y, z, "YZ")

    assert len(results_list) == len(transform_features)*3
    return results_list


def compute_dataset_features(path, skip_samples, limit_samples):

    import npyfile

    with npyfile.Reader(path) as data:

        # Check that data is expected format
        # 
        shape = data.shape
        assert len(shape) == 3, shape
        n_samples, window_length, n_axes = shape
        assert n_axes == 3, shape
        assert window_length == 128, shape

        # FIXME: use single precision float, or h/int16
        #assert data.typecode == 'f', data.typecode
        #assert data.itemsize == 1, data.itemsize

        # pre-allocate values
        x_values = array.array('f', (0 for _ in range(window_length)))
        y_values = array.array('f', (0 for _ in range(window_length)))
        z_values = array.array('f', (0 for _ in range(window_length)))

        chunk_size = window_length*n_axes
        sample_counter = 0

        data_chunks = data.read_data_chunks(chunk_size, offset=chunk_size*skip_samples)
        for arr in data_chunks:

            # process the data
            # De-interleave data from XYZ1 XYZ2... into separate continious X,Y,Z
            for i in range(window_length):
                x_values[i] = arr[(i*3)+0]
                y_values[i] = arr[(i*3)+1]
                z_values[i] = arr[(i*3)+2]

            assert len(x_values) == window_length
            assert len(y_values) == window_length
            assert len(z_values) == window_length
            features = calculate_features_xyz((x_values, y_values, z_values))
            yield features

            sample_counter += 1
            if sample_counter > limit_samples:
                break

def main():

    path = 'pamap2_windows.npy'
    skip_samples = 0
    limit_samples = 2

    generator = compute_dataset_features(path, skip_samples=skip_samples, limit_samples=2)
    for features in generator:
        print('features', features)


if __name__ == '__main__':
    main()

