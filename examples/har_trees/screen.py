# Example program for the T-WATCH S3

import random

from machine import Pin, SPI
import st7789_base, st7789_ext
import time
from axp2101 import AXP2101

def main():
    # Setup the PMU chip.
    twatch_pmu = AXP2101()
    twatch_pmu.twatch_s3_poweron()
    print("[AXP2101] Battery voltage is", twatch_pmu.get_battery_voltage())

    # Power on the display backlight.
    bl = Pin(45,Pin.OUT)
    bl.on()

    # Our display does not have a MISO pin, but the MicroPython
    # SPI implementation does not allow to avoid specifying one, so
    # we use just a not used pin in the device.
    display = st7789_ext.ST7789(
        SPI(1, baudrate=40000000, phase=0, polarity=1, sck=18, miso=37),
        240, 240,
        reset=False,
        dc=Pin(38, Pin.OUT),
        cs=Pin(12, Pin.OUT),
    )
    display.init(landscape=False,mirror_y=True,mirror_x=True,inversion=True)
    display.enable_framebuffer()
  

    print("displaying random colors")
    while True:
        start = time.ticks_ms()
        display.fb.fill(
            display.fb_color(
                random.getrandbits(8),
                random.getrandbits(8),
                random.getrandbits(8),
            ),
        )
        for i in range(250):
            display.fb.pixel(random.randint(0,240),
                             random.randint(0,240),
                             display.fb_color(255,255,255))

        display.show()
        elapsed = time.ticks_ms() - start
        print("Ticks per screen fill:", elapsed)

        # Pause 2 seconds.
        time.sleep(2)

main()
