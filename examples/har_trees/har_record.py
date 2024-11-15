

from machine import Pin, I2C
from mpu6886 import MPU6886
import npyfile

import time
import array
import struct

def format_time(secs):
    year, month, day, hour, minute, second, _, _ = time.gmtime(secs)
    formatted = f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}'
    return formatted

classes = [
    'jumpingjack',
    'lunge',
    'squat',
    'pushup',
]

samplerate = 100
chunk_length = 50
file_duration = 10.0

def decode_samples(buf : bytearray, samples : array.array, bytes_per_sample):
    """
    Convert raw bytes into int16 arrays
    """
    assert (len(buf) % bytes_per_sample) == 0
    n_samples = len(buf) // bytes_per_sample
    assert len(samples) == 3*n_samples, (len(samples), 3*n_samples)

    #view = memoryview(buf)
    for i in range(n_samples):
        # NOTE: temperature (follows z) is ignored
        x, y, z = struct.unpack_from('>hhh', buf, i*bytes_per_sample)
        samples[(i*3)+0] = x
        samples[(i*3)+1] = y
        samples[(i*3)+2] = z


def main():
    mpu = MPU6886(I2C(0, sda=21, scl=22, freq=100000))

    # Enable FIFO at a fixed samplerate
    mpu.fifo_enable(True)
    mpu.set_odr(samplerate)

    chunk = bytearray(mpu.bytes_per_sample*chunk_length)
    decoded = array.array('h', (0 for _ in range(3*chunk_length)))
    recording_file = None
    recording_samples = int(file_duration * samplerate)

    # TODO: support start/stop of recording
    recording = True

    # TODO: support specifying class being recorded
    current_class = classes[0]

    start_time = time.ticks_ms()

    while True:

        try:

            # always read data from FIFO, to avoid overflow
            count = mpu.get_fifo_count()
            if count >= chunk_length:
                start = time.ticks_ms()
                mpu.read_samples_into(chunk)
                decode_samples(chunk, decoded, mpu.bytes_per_sample)
                t = (time.ticks_ms() - start_time)/1000.0

                if recording:
                    # store the data
                    if recording_file is None:
                        time_str = format_time(time.time())
                        out_path = f'har_{time_str}_{current_class}.npy'
                        out_typecode = 'h'
                        out_shape = (3, recording_samples)
                        recording_file = npyfile.Writer(open(out_path, 'w'), out_shape, out_typecode)
                        print(f'record-file-open t={t:.3f} file={out_path}')

                    recording_file.write_values(decoded)
                    print(f'record-chunk t={t:.3f}')
                    if recording_file.written_bytes > 3*2*recording_samples:
                        # rotate file
                        recording_file.close()
                        recording_file = None
                        print(f'record-file-rotate t={t:.3f}')

            time.sleep_ms(1)

        except Exception as e:
            if recording_file is not None:
                recording_file.close()
                recording_file = None
            raise e


if __name__ == '__main__':
    main()
