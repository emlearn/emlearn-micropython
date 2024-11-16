

import machine
from machine import Pin, I2C

from mpu6886 import MPU6886

# mpremote mip install "github:peterhinch/micropython-async/v3/primitives"
from primitives import Pushbutton

import asyncio
import os
import time
import array
import struct
import gc

from recorder import Recorder

# for display
# mpremote mip install "github:peterhinch/micropython-nano-gui"
from color_setup import ssd
from gui.core.writer import Writer
from gui.core.nanogui import refresh
from gui.widgets.meter import Meter
from gui.widgets.label import Label
import gui.fonts.courier20 as fixed

# Cleanup after import frees considerable memory
gc.collect()

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


def render_display(selected_class):
    start_time = time.ticks_ms()
   
    ssd.fill(0)

    Writer.set_textpos(ssd, 0, 0)  # In case previous tests have altered it
    wri = Writer(ssd, fixed, verbose=False)
    wri.set_clip(False, False, False)

    textfield = Label(wri, 10, 20, wri.stringlen(selected_class))
    textfield.value(selected_class)

    refresh(ssd)

    duration = time.ticks_ms() - start_time
    print('render-display-done', duration)



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
        # FIXME: change suffix on the recorder
        render_display(c)

        print(f'har-record-cycle class={c}')

    button_pin = machine.Pin(37, machine.Pin.IN, machine.Pin.PULL_UP) # Button A on M5StickC PLUS2
    button = Pushbutton(button_pin)
    button.long_func(on_longpress, args=())
    button.double_func(on_doubleclick, args=())

    async def read_data():

        # UNCOMMENT to clean up data_dir
        recorder.delete()
    
        render_display(classes[class_selected])
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

            await asyncio.sleep(0.10)

    with Recorder(samplerate, file_duration, directory=data_dir) as recorder:

        asyncio.run(read_data())

if __name__ == '__main__':
    main()
