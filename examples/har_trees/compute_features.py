
import sys

import npyfile

from timebased import calculate_features_xyz

def compute_dataset_features(data: npyfile.Reader, skip_samples=0, limit_samples=None):

    # Check that data is expected format
    shape = data.shape
    assert len(shape) == 3, shape
    n_samples, window_length, n_axes = shape
    assert n_axes == 3, shape
    assert window_length == 128, shape

    # We expect data to be h/int16
    assert data.typecode == DATA_TYPECODE, data.typecode
    assert data.itemsize == 2, data.itemsize

    # pre-allocate values
    x_values = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    y_values = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))
    z_values = array.array(DATA_TYPECODE, (0 for _ in range(window_length)))

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

        #print(x_values)
        #print(y_values)
        #print(z_values)

        assert len(x_values) == window_length
        assert len(y_values) == window_length
        assert len(z_values) == window_length

        feature_calc_start = time.ticks_ms()
        features = calculate_features_xyz((x_values, y_values, z_values))
        duration = time.ticks_diff(time.ticks_ms(), feature_calc_start)
        print('feature-calc-end', duration)

        yield features

        sample_counter += 1
        if limit_samples is not None and sample_counter > limit_samples:
            break

def main():

    if len(sys.argv) != 3:

    _, inp, out = sys.argv

    skip_samples = 0
    limit_samples = 

    # FIXME: open and write output
    with npyfile.Reader(path) as data:

        generator = compute_dataset_features(data, skip_samples=skip_samples, limit_samples=limit_samples)
        for features in generator:
            print('features', len(features), features)


if __name__ == '__main__':
    main()

