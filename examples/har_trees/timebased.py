

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
import time

#########################################

DATA_TYPECODE = 'h' # int16

L1_NORM = 1
L2_NORM = 2
L2_NORM_SQUARED = 3

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

N_FEATURES = 92

@micropython.native
def l2_sum(arr):
    acc = 0.0
    for x in arr:
        acc += (x*x)       
    return acc

@micropython.native
def ordered_features(matrix : array.array, results : array.array):
    assert len(results) == ORDERED_FEATURES_N

    WINDOW_SIZE = len(matrix)
    v = matrix
    MEDIAN = WINDOW_SIZE // 2
    Q1 = WINDOW_SIZE // 4
    Q3 = 3 * WINDOW_SIZE // 4

    l = sorted(v)
    sm = sum(l)
    sqs = l2_sum(l)
    avg = sm / len(l)
    median = l[MEDIAN]  
    mn = l[0]
    mx = l[-1]
    energy = ((sqs / len(l)) ** 0.5) # rms
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

def corr(a, b, out):

    for i, (v1, v2) in enumerate(zip(a, b)):
        cc = np.corrcoef(v1, v2)
        r = cc[0][1]
        if math.isnan(r):
            r = 1.0 # std == 0; assume perfect correlation (wise?)
        out[i] = r

#########################################w

@micropython.native
def jerk_filter(inp, out):
    out[0] = 0
    for i in range(len(inp) - 1):
        out[i] = (inp[i + 1] - inp[i])

#########################################w

@micropython.native
def norm_filter_l1(x, y, z, out):
    for i in range(len(x)):
        out[i] = abs(x[i]) + abs(y[i]) + abs(z[i])

@micropython.native
def norm_filter_l2(x, y, z, out):
    for i in range(len(x)):
        out[i] = (x[i]*x[i] + y[i]*y[i] + z[i]*z[i])**0.5

@micropython.native
def norm_filter_l2_squared(x, y, z, out):
    for i in range(len(x)):
        out[i] = (x[i]*x[i] + y[i]*y[i] + z[i]*z[i])

##########################################

@micropython.native
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

@micropython.native
def median_filter(inp, out):

    #return None

    out[0] = inp[0]
    for i in range(1, len(inp) - 1):
        v = median(inp[i-1], inp[i], inp[i+1])
        out[i] = v
    out[-1] = inp[-1]


def calculate_features_xyz(xyz):

    xo, yo, zo = xyz
    window_length = len(xo)

    # pre-allocate arrays
    alloc_start = time.ticks_ms()

    xm = array.array(DATA_TYPECODE, range(window_length))
    ym = array.array(DATA_TYPECODE, range(window_length))
    zm = array.array(DATA_TYPECODE, range(window_length))

    l2_norm_sq = array.array(DATA_TYPECODE, range(window_length))
    l1_norm = array.array(DATA_TYPECODE, range(window_length))
    l1_norm_jerk = array.array(DATA_TYPECODE, range(window_length))
    l2_norm_sq_jerk = array.array(DATA_TYPECODE, range(window_length))

    # Compute filtered versions
    filter_start = time.ticks_ms()

    median_filter(xo, xm)
    median_filter(yo, ym)
    median_filter(zo, zm)

    # Doing a derivative is going to reduce the effective recovery data frequency 2 times.
    # This assumes that the data is already low-pass filtered (for 50 Hz to 20 Hz in the dataset)
    # therefore high-frequency components are negligible
    jerk_filter(xm, xo)
    jerk_filter(ym, yo)
    jerk_filter(zm, zo)

    x = xm
    y = ym
    z = zm
    x_jerk = xo
    y_jerk = yo
    z_jerk = zo

    # compute norms
    # do the squared L2 norm for now instead of the normal L2 norm
    norm_filter_l1(x, y, z, l1_norm)
    norm_filter_l2_squared(x, y, z, l2_norm_sq)
    jerk_filter(l1_norm, l1_norm_jerk)
    jerk_filter(l2_norm_sq, l2_norm_sq_jerk)

    filter_end = time.ticks_ms()

    norm_options = [None, L1_NORM, L2_NORM_SQUARED]
    jerk_options = [False, True]

    all_results = []

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

    features_end = time.ticks_ms()
    alloc_duration = time.ticks_diff(filter_start, alloc_start) 
    filter_duration = time.ticks_diff(filter_end, filter_start)
    feature_duration = time.ticks_diff(features_end, filter_end)

    #print('feature-calc-details', alloc_duration, filter_duration, feature_duration)
    assert len(all_results) == N_FEATURES, len(all_results)

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
#norm_features = []

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
#transform_features = []

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


