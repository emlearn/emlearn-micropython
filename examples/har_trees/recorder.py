
import npyfile

# mpremote mip install "github:peterhinch/micropython-async/v3/primitives"
from primitives import Pushbutton

import asyncio
import os
import time
import array
import struct
import gc


def format_time(secs):
    year, month, day, hour, minute, second, _, _ = time.gmtime(secs)
    formatted = f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}'
    return formatted

def file_or_dir_exists(filename):
    try:
        os.stat(filename)
        return True
    except OSError:
        return False

class Recorder():
    """Record accelerometer data to .npy files"""
    
    def __init__(self, samplerate, duration, directory='recorder_data', suffix='.npy'):
        # config      
        self._directory = directory
        assert directory[-1] != '/'
        self._suffix  = suffix
        self._recording_samples = int(duration * samplerate)

        # state
        self._recording_file = None
        self._recording = False

        if not file_or_dir_exists(self._directory):
            os.mkdir(self._directory)

    def start(self):
        self._recording = True
        print('recorder-start')

    def stop(self):
        self.close()
        self._recording = False
        print('recorder-stop')

    def process(self, data):

        if not self._recording:
            return

        t = time.ticks_ms()/1000.0

        if self._recording_file is None:
            # open file
            time_str = format_time(time.time())
            out_path = f'{self._directory}/{time_str}_{self._suffix}'
            out_typecode = 'h'
            out_shape = (3, self._recording_samples)
            self._recording_file = npyfile.Writer(open(out_path, 'w'), out_shape, out_typecode)
            print(f'record-file-open t={t:.3f} file={out_path}')

        # TODO: avoid writing too much at end of file
        self._recording_file.write_values(data)
        print(f'recorder-write-chunk t={t:.3f}')
        if self._recording_file.written_bytes > 3*2*self._recording_samples:
            # rotate file
            self.close()
            print(f'record-file-rotate t={t:.3f}')

    def delete(self):
        for f in os.listdir(self._directory):
            p = self._directory + '/' + f
            print('recorder-delete-file', p)
            os.unlink(p)

    def close(self):
        if self._recording_file is not None:
            self._recording_file.close()
            self._recording_file = None

    # Support working as a context manager, to automatically clean up
    def __enter__(self):
        pass      
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
