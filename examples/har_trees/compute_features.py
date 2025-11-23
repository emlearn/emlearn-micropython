
import sys
import array
import time

import npyfile

from timebased import calculate_features_xyz, DATA_TYPECODE, N_FEATURES

def compute_dataset_features(data: npyfile.Reader, window_length,
        hop_length=None,
        skip_samples=0, limit_samples=None, verbose=0):

    if hop_length is None:
        hop_length = window_length

    if window_length % hop_length != 0:
        raise ValueError(f"hop_length must be an even divisor of window_length. Got window={window_length} hop={hop_length}")


    # Check that data is expected format
    shape = data.shape
    assert len(shape) == 2, shape
    n_samples, n_axes = shape
    assert n_axes == 3, shape

    # We expect data to be h/int16
    assert data.typecode == DATA_TYPECODE, data.typecode
    assert data.itemsize == 2, data.itemsize

    # pre-allocate values
    x_values = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    y_values = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    z_values = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))

    chunk_size = hop_length*n_axes
    window_counter = 0
    start_idx = 0

    data_chunks = data.read_data_chunks(chunk_size, offset=chunk_size*skip_samples)

    for arr in data_chunks:

        print('cc', len(arr))
        if len(arr) < chunk_size:
            # short read, last data piece, ignore
            continue

        # Window was full, make room for more
        if start_idx >= window_length:
            overlap = window_length - hop_length
            if overlap > 0:
                x_values[:overlap] = x_values[hop_length:]
                y_values[:overlap] = y_values[hop_length:]
                z_values[:overlap] = z_values[hop_length:]
            start_idx = overlap

        # Copy the input data
        # De-interleave data from XYZ1 XYZ2... into separate continious X,Y,Z
        for i in range(hop_length):
            x_values[i] = arr[(i*3)+0]
            y_values[i] = arr[(i*3)+1]
            z_values[i] = arr[(i*3)+2]
        start_idx += hop_length

        # waiting for window to fill
        if start_idx < window_length:
            continue

        feature_calc_start = time.ticks_ms()
        features = calculate_features_xyz((x_values, y_values, z_values))
        duration = time.ticks_diff(time.ticks_ms(), feature_calc_start)
        if verbose > 2:
            print('feature-calc-end', duration)

        yield features
        window_counter += 1

        if limit_samples is not None and window_counter > limit_samples:
            break

def parse():
    import argparse

    parser = argparse.ArgumentParser(description='Compute features from NPY file')
    parser.add_argument('--input', required=True, help='Input .npy file')
    parser.add_argument('--output', required=True, help='Output .npy file')
    parser.add_argument('--samplerate', type=int, default=None, help='Samplerate. Currently ignored')
    parser.add_argument('--skip', type=int, default=0, help='Number of samples to skip (default: 0)')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of samples to process (default: None)')
    parser.add_argument('--window_length', type=int, default=128, help='Window length (default: 128)')
    parser.add_argument('--hop_length', type=int, default=None, help='Hop length (default: window_length)')

    args = parser.parse_args()
    return args

def main():

    args = parse()

    out_typecode = 'f'
    n_features = N_FEATURES    
    features_array = array.array(out_typecode, (0 for _ in range(n_features)))

    with npyfile.Reader(args.input) as data:
        n_samples, n_axes = data.shape

        n_windows = (n_samples - args.window_length) // args.hop_length

        out_shape = (n_windows, n_features)
        with npyfile.Writer(args.output, out_shape, out_typecode) as out:

            generator = compute_dataset_features(data,
                window_length=args.window_length,
                hop_length=args.hop_length,
                skip_samples=args.skip,
                limit_samples=args.limit,
            )
            for features in generator:
                #print('features', len(features), features)
                assert len(features) == n_features, (len(features), n_features)

                for i, f in enumerate(features):
                    features_array[i] = f

                out.write_values(features_array)

if __name__ == '__main__':
    main()

