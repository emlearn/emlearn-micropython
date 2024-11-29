
import array
import math

import npyfile

print(npyfile.__file__)

from recorder import Recorder

samplerate = 100
chunk_length = 30
file_duration = 2.0
test_data_dir = 'test_har_record'

def check_file(path):
    shape, data = npyfile.load(path)
    return shape

def test_recorder_one_file():

    decoded = array.array('h', (0 for _ in range(3*chunk_length)))

    n_chunks = math.ceil((file_duration * samplerate) / chunk_length)

    with Recorder(samplerate, file_duration, directory=test_data_dir) as recorder:  
        recorder.start()

        for i in range(n_chunks):
            recorder.process(decoded)

        p = recorder._recording_file_path

        shape = check_file(p)
        assert shape[0] == 3
        assert shape[1] == int(file_duration*samplerate)

if __name__ == '__main__':
    test_recorder_one_file()
