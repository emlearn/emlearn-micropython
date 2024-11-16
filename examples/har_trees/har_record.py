

import machine
from machine import Pin, I2C
from mpu6886 import MPU6886
import npyfile

# mpremote mip install "github:peterhinch/micropython-async/v3/primitives"
from primitives import Pushbutton

import asyncio
import os
import time
import array
import struct
import gc

# Cleanup after import frees considerable memory
gc.collect()

def format_time(secs):
    year, month, day, hour, minute, second, _, _ = time.gmtime(secs)
    formatted = f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}'
    return formatted

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

# Configuration
classes = [
    'jumpingjack',
    'lunge',
    'squat',
    'pushup',
]

samplerate = 100
chunk_length = 50
file_duration = 5.0
data_dir = 'har_record'


def main():
    mpu = MPU6886(I2C(0, sda=21, scl=22, freq=100000))

    # Enable FIFO at a fixed samplerate
    mpu.fifo_enable(True)
    mpu.set_odr(samplerate)

    chunk = bytearray(mpu.bytes_per_sample*chunk_length) # raw bytes
    decoded = array.array('h', (0 for _ in range(3*chunk_length))) # decoded int16

    # Internal LED on M5StickC PLUS2
    led_pin = machine.Pin(19, machine.Pin.OUT)

    # On M5StickC we need to set HOLD pin to stay alive when on battery
    hold_pin = machine.Pin(4, machine.Pin.OUT)
    hold_pin.value(1)

    # Support cycling between classes, to indicate which is being recorded
    class_selected = 0

    def on_longpress():
        # toggle recording state
        if recorder._recording:
            recorder.stop()
        else:
            recorder.start()

    def on_doubleclick():
        # cycle between selected class
        nonlocal class_selected
        class_selected += 1
        if class_selected >= len(classes):
            class_selected = 0
        c = classes[class_selected]
        print(f'har-record-cycle class={c}')

    button_pin = machine.Pin(37, machine.Pin.IN, machine.Pin.PULL_UP) # Button A on M5StickC PLUS2
    button = Pushbutton(button_pin)
    button.long_func(on_longpress, args=())
    button.double_func(on_doubleclick, args=())

    async def read_data():

        # UNCOMMENT to clean up data_dir
        recorder.delete()
    
        print('har-record-ready')

        while True:
        
            # always read data from FIFO, to avoid overflow
            count = mpu.get_fifo_count()
            if count >= chunk_length:
                start = time.ticks_ms()
                mpu.read_samples_into(chunk)
                decode_samples(chunk, decoded, mpu.bytes_per_sample)

                # record data (if enabled)
                recorder.process(decoded)

            # Let LED reflect recording state        
            led_pin.value(1 if recorder._recording else 0)

            await asyncio.sleep(0.050)

    with Recorder(samplerate, file_duration, directory=data_dir) as recorder:

        asyncio.run(read_data())

if __name__ == '__main__':
    main()
