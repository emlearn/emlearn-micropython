import os
import subprocess
import tempfile

import pandas
import numpy


def convert_to_raw(df: pandas.DataFrame,
        in_max=60.0,
        out_max=(2**15)-1,
        dtype=numpy.int16):

    missing = df.isna().any().sum() / len(df)
    print('missing values', 100*missing, '%')
    df = df.fillna(0.0)
    scaled = ((df / in_max) * out_max).clip(-out_max, out_max)
    out = scaled.astype(dtype)

    return out

def make_label_track(times, events,
    label_column='label', default=None):

    out = pandas.Series(numpy.repeat(default, len(times)), index=times)
    for idx, event in events.iterrows():
        s = event['start_time']
        e = event['end_time']
        out.loc[s:e] = event[label_column]
    
    return out

def process_data(df : pandas.DataFrame,
        micropython_bin='micropython',
        samplerate = 100,
        window_length = 256,
        fft_start = 0,
        fft_end = 16
    ):

    fft_features = fft_end - fft_start

    here = os.path.dirname(__file__)

    hop_length = window_length
    time_resolution = pandas.Timedelta(seconds=1)/samplerate * hop_length
    print(time_resolution)

    # check input
    data = numpy.ascontiguousarray(df.values)    
    assert len(data.shape) == 2, data.shape
    assert data.shape[1] == 3, data.shape
    assert data.dtype == numpy.int16, data.dtype
    assert data.flags['C_CONTIGUOUS'], data.flags

    times = df.index
    
    with tempfile.TemporaryDirectory() as temp:
        temp = ''
        in_path = os.path.join(temp, 'input.npy')
        out_path = os.path.join(temp, 'output.npy')

        # write input data
        numpy.save(in_path, data, allow_pickle=False)

        # run the program, as subprocess
        script = os.path.join(here, 'example.py')
        args = [
            micropython_bin,
            script,
            in_path,
            out_path,
        ]
        cmd = ' '.join(args)
        print('run-cmd', cmd)
        subprocess.check_output(args)

        # load output data
        assert os.path.exists(out_path)
        out = numpy.load(out_path)

        fft_columns = [ f'fft.{i}' for i in range(fft_features) ]
        columns = ['peak2peak', 'fft_energy'] + fft_columns
        df = pandas.DataFrame(out, columns=columns)

        df['time'] = times.min() + (time_resolution * numpy.arange(0, len(df)))
        df = df.set_index('time')
        return df
