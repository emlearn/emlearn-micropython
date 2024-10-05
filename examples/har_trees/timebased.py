

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

MAGNITUDE = L2_NORM
MAGNITUDE_SQUARED = L2_NORM_SQUARED

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

def ordered_features(results, matrix, axis, is_all=False):
    WINDOW_SIZE = len(matrix)
    v = matrix
    MEDIAN = WINDOW_SIZE // 2
    Q1 = WINDOW_SIZE // 4
    Q3 = 3 * WINDOW_SIZE // 4
    mean = []
    iqr = []
    mn = []
    mx = []
    std = []
    energy = []
    entropy = []
    median = []
    q25 = []
    q75 = []

    l = sorted(list(v))
    l2 = [x*x for x in l]
    sm = sum(l)
    sqs = sum(l2)
    avg = sum(l) / len(l)

    mean = avg
    median = l[MEDIAN]
    q25 = l[Q1]
    q75 = l[Q3]
    iqr = (l[Q3] - l[Q1])
   
    mn = l[0]
    mx = l[-1]

    energy = ((sqs / len(l2)) ** 0.5) # rms
    std = ((sqs - avg * avg) ** 0.5)

        #mad_list = [abs(x - l[MEDIAN]) for x in l]
        #mad_list.sort()
        #mad.append(mad_list[MEDIAN])
        #bins, bin_edges = np.histogram(l, bins=10, density=True)
        #print(scipy.stats.entropy(bins), bins)
        # scipy.stats.entropy(bins)
    entropy = 0 # FIXME: not implemented

    if len(axis) and axis[0] != "-":
        axis = "-" + axis
    alltxt = "all" if is_all else ""
    results["tTotalAcc-mean{}(){}".format(alltxt, axis)] = mean
    results["tTotalAcc-median{}(){}".format(alltxt, axis)] = median
    results["tTotalAcc-q25{}(){}".format(alltxt, axis)] = q25
    results["tTotalAcc-q75{}(){}".format(alltxt, axis)] = q75
    results["tTotalAcc-iqr{}(){}".format(alltxt, axis)] = iqr
    results["tTotalAcc-min{}(){}".format(alltxt, axis)] = mn
    results["tTotalAcc-max{}(){}".format(alltxt, axis)] = mx
    results["tTotalAcc-std{}(){}".format(alltxt, axis)] = std
    results["tTotalAcc-energy{}(){}".format(alltxt, axis)] = energy
    results["tTotalAcc-entropy{}(){}".format(alltxt, axis)] = entropy
    #results["tTotalAcc-mad()" + axis] = mad

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

#########################################

def areg(results, matrix, suffix):
    from spectrum import arburg

    corrs = [[] for _ in range(4)]
    for v in matrix:
        l = sorted(list(v))
        a = arburg(l, 4)
        for i in range(4):
            corrs[i].append(np.real(a[0][i]))

    if len(suffix):
        suffix += ","
    for i in range(4):
        results["tTotalAcc-arCoeff()-" + suffix + str(i + 1)] = corrs[i]

#########################################w

def jerk_filter(matrix):
    row = matrix

    jrow = [0]
    for i in range(len(row) - 1):
        jrow.append(row[i + 1] - row[i])
    return jrow

#########################################w

def norm(x, y, z, code):
    if code == L1_NORM:
        return [abs(x[i]) + abs(y[i]) + abs(z[i]) for i in range(len(x))]
    if code == L2_NORM:
        return [(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])**0.5 for i in range(len(x))]
    # L2_NORM_SQUARED
    return [(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) for i in range(len(x))]

def norm_filter(x, y, z, code):
    return norm(x, y, z, code)

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

    for do_jerk in jerk_options:
        if do_jerk:
            tx = x_jerk; ty = y_jerk; tz = z_jerk
            tl1 = l1_norm_jerk
            tl2 = l2_norm_sq_jerk
            jerk_name = "Jerk"
        else:
            tx = x; ty = y; tz = z
            tl1 = l1_norm
            tl2 = l2_norm_sq
            jerk_name = ""

        results = calculate_features_of_transform(tx, ty, tz, jerk_name)
        all_results += results

        results = calculate_features_of_norm_transform(tl1, jerk_name, "L1Norm")
        all_results += results

        results = calculate_features_of_norm_transform(tl2, jerk_name, "MagSq")

        all_results += results

    assert len(all_results) == 92

    return all_results

norm_transform_names = [
    "tTotalAcc-mean()",
    "tTotalAcc-min()",
    "tTotalAcc-max()",
    "tTotalAcc-median()",           
    "tTotalAcc-iqr()",
    "tTotalAcc-energy()",
    "tTotalAcc-std()",
    # skip the autoregression
    "tTotalAcc-entropy()",
]

def calculate_features_of_norm_transform(m, jerk_name, norm_name):
    results = {}
    ordered_features(results, m, "")

    results_list = []
    for n in norm_transform_names:
        results_list.append(results[n])

    suffix = jerk_name + norm_name
    assert len(results_list) == len(norm_transform_names)

    return results_list


transform_names = [
    "tTotalAcc-mean()-X",
    "tTotalAcc-mean()-Y",
    "tTotalAcc-mean()-Z",
    #"tTotalAcc-meanall()",
    "tTotalAcc-max()-X",
    "tTotalAcc-max()-Y",
    "tTotalAcc-max()-Z",
    #"tTotalAcc-maxall()",
    "tTotalAcc-min()-X",
    "tTotalAcc-min()-Y",
    "tTotalAcc-min()-Z",
    #"tTotalAcc-minall()",
    "tTotalAcc-median()-X",
    "tTotalAcc-median()-Y",
    "tTotalAcc-median()-Z",
    #"tTotalAcc-medianall()",
    "tTotalAcc-q25()-X",
    "tTotalAcc-q25()-Y",
    "tTotalAcc-q25()-Z",
    "tTotalAcc-q75()-X",
    "tTotalAcc-q75()-Y",
    "tTotalAcc-q75()-Z",
    "tTotalAcc-iqr()-X",
    "tTotalAcc-iqr()-Y",
    "tTotalAcc-iqr()-Z",
    #"tTotalAcc-iqrall()",
    "tTotalAcc-energy()-X",
    "tTotalAcc-energy()-Y",
    "tTotalAcc-energy()-Z",
    # do not do "all" for energy, std, and entropy - if these things vary,
    # they should show up as high energy/std/entropy of the magnitude???
    "tTotalAcc-std()-X",
    "tTotalAcc-std()-Y",
    "tTotalAcc-std()-Z",
    #"tTotalAcc-stdall()",
    #"tTotalAcc-correlation()-XY",
    #"tTotalAcc-correlation()-XZ",
    #"tTotalAcc-correlation()-YZ",
    "tTotalAcc-entropy()-X",
    "tTotalAcc-entropy()-Y",
    "tTotalAcc-entropy()-Z",
    #"tTotalAcc-sma()",
]

def calculate_features_of_transform(x, y, z, jerk_name):
    results = {}

    ordered_features(results, x, "X")
    ordered_features(results, y, "Y")
    ordered_features(results, z, "Z")

    ordered_features(results, x + y + z, "", True)

    # FIXME: raises exception
    #corr(results, x, y, "XY")
    #corr(results, x, z, "XZ")
    #corr(results, y, z, "YZ")

    results_list = []
    for n in transform_names:
        results_list.append(results[n])

    suffix = jerk_name

    assert len(results_list) == len(transform_names)
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

            # if it was stored as continious runs of XX...YY...ZZ... instead
            #x = arr[:window_length]
            #y = arr[window_length:window_length*2]
            #z = arr[window_length*2:]

            assert len(x_values) == window_length
            assert len(y_values) == window_length
            assert len(z_values) == window_length
            # FIXME: do not include feature names in each computation
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

