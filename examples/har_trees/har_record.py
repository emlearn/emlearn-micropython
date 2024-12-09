

import machine
from machine import Pin, I2C

#from mpu6886 import MPU6886

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
#from color_setup import ssd
#from gui.core.writer import Writer
#from gui.core.nanogui import refresh
#from gui.widgets.meter import Meter
#from gui.widgets.label import Label
#import gui.fonts.courier20 as fixed

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


def render_display(selected_class, recording : bool):
    start_time = time.ticks_ms()
   
    ssd.fill(0)

    Writer.set_textpos(ssd, 0, 0)  # In case previous tests have altered it
    wri = Writer(ssd, fixed, verbose=False)
    wri.set_clip(False, False, False)

    text = f'Activity:'
    text1 = Label(wri, 10, 10, wri.stringlen(text))
    text1.value(text)

    text2 = Label(wri, 30, 10, wri.stringlen(selected_class))
    text2.value(selected_class)

    state = '[recording]' if recording else '[ready]'
    text3 = Label(wri, 80, 10, wri.stringlen(state))
    text3.value(state)

    refresh(ssd)

    duration = time.ticks_ms() - start_time
    print('render-display-done', duration)



# Configuration
classes = [
    'jumpingjack',
    'lunge',
    'squat',
    'other',
]

samplerate = 100
chunk_length = 50
file_duration = 10.0
data_dir = 'har_record'


def main():
    #mpu = MPU6886(I2C(0, sda=21, scl=22, freq=100000))

    from machine import SPI

    # mpremote mip install https://raw.githubusercontent.com/antirez/t-watch-s3-micropython/refs/heads/main/axp2101.py
    from axp2101 import AXP2101
    pmu = AXP2101()
    pmu.twatch_s3_poweron()

    # mpremote mip install https://raw.githubusercontent.com/antirez/t-watch-s3-micropython/refs/heads/main/st7789_base.py
    # mpremote mip install https://raw.githubusercontent.com/antirez/t-watch-s3-micropython/refs/heads/main/st7789_ext.py
    import st7789_base, st7789_ext

    # Setup the PMU chip & turn on the backlight.
    bl = Pin(45,Pin.OUT)
    bl.on()

    from machine import SoftSPI

    # XXX: for some reason hardware SPI does not work?
    #spi = SPI(1, baudrate=10_000_000, phase=0, polarity=1, sck=Pin(18), mosi=Pin(13), miso=Pin(37))
    
    spi = SoftSPI(baudrate=10_000_000, phase=0, polarity=1, sck=Pin(18), mosi=Pin(13), miso=Pin(37))

    # Setup TFT.
    display = st7789_ext.ST7789(
        spi,
        240, 240,
        reset=False,
        dc=Pin(38, Pin.OUT, value=0),
        cs=Pin(12, Pin.OUT, value=1),
    )
    display.init(landscape=False,mirror_y=True,mirror_x=True,inversion=True)

    # Enable FIFO at a fixed samplerate
    #mpu.fifo_enable(True)
    #mpu.set_odr(samplerate)

    #chunk = bytearray(mpu.bytes_per_sample*chunk_length) # raw bytes
    #decoded = array.array('h', (0 for _ in range(3*chunk_length))) # decoded int16

    # Internal LED on M5StickC PLUS2
    #led_pin = machine.Pin(19, machine.Pin.OUT)

    # On M5StickC we need to set HOLD pin to stay alive when on battery
    #hold_pin = machine.Pin(4, machine.Pin.OUT)
    #hold_pin.value(1)

    # Support cycling between classes, to indicate which is being recorded
    class_selected = 0

    display.enable_framebuffer()

    while True:
        #display.fb.fill(display.fb_color(100,0,0))
        display.fb.text("Hello world",50,50,display.fb_color(255,255,255))
        display.show()
        print('display')
        time.sleep(1)

    return



    def update_display():
        c = classes[class_selected]
        render_display(c, recorder._recording)
        led_pin.value(1 if recorder._recording else 0)

    def on_longpress():
        # toggle recording state
        if recorder._recording:
            recorder.stop()
        else:
            recorder.start()
        update_display()

    def on_doubleclick():
        # cycle between selected class
        nonlocal class_selected
        class_selected += 1
        if class_selected >= len(classes):
            class_selected = 0
        c = classes[class_selected]
        recorder.set_class(c)
        update_display()

        print(f'har-record-cycle class={c}')

    #button_pin = machine.Pin(37, machine.Pin.IN, machine.Pin.PULL_UP) # Button A on M5StickC PLUS2
    #button = Pushbutton(button_pin)
    #button.long_func(on_longpress, args=())
    #button.double_func(on_doubleclick, args=())

    async def update_display_loop():
        # workaround for display not always updating on boot
        while True:
            update_display()
            await asyncio.sleep(1.0)

    async def read_data():
        while True:        
            await asyncio.sleep(1.00)

    async def run():
        await asyncio.gather(update_display_loop(), read_data())
    
    with Recorder(samplerate, file_duration, directory=data_dir) as recorder:

        # UNCOMMENT to clean up data_dir
        #recorder.delete()

        print('har-record-ready')

        asyncio.run(run())

if __name__ == '__main__':
    main()
