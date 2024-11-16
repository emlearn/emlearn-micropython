
from machine import Pin, I2C
from mpu6886 import MPU6886

import time
import struct
import array

#from windower import TriaxialWindower, empty_array
#import timebased
#import emlearn_trees    

# mpremote mip install "github:peterhinch/micropython-nano-gui/drivers/st7789"
from color_setup import ssd

# mpremote mip install "github:peterhinch/micropython-nano-gui"
# On a monochrome display Writer is more efficient than CWriter.
from gui.core.writer import Writer
from gui.core.nanogui import refresh
from gui.widgets.meter import Meter
from gui.widgets.label import Label

# Fonts
import gui.fonts.courier20 as fixed
#import gui.fonts.font6 as small


def render_display(db : float):
    start_time = time.ticks_ms()
    
    ssd.fill(0)
    #refresh(ssd)

    Writer.set_textpos(ssd, 0, 0)  # In case previous tests have altered it
    wri = Writer(ssd, fixed, verbose=False)
    wri.set_clip(False, False, False)

    warn_text = 'Loud!'
    numfield = Label(wri, 5, 0, wri.stringlen('99.9'))
    textfield = Label(wri, 40, 34, wri.stringlen(warn_text))

    numfield.value('{:5.1f} dBa'.format(db))

    if db > 75.0:
        textfield.value(warn_text, True)
    else:
        textfield.value('')

    refresh(ssd)

    duration = time.ticks_ms() - start_time
    print('render-display-done', duration)


def main():

    next_display_update = 0.0

    while True:
        if time.time() >= next_display_update:
            soundlevel_db = 10.0
            if soundlevel_db is not None:
                render_display(db=soundlevel_db)
            last_display_update = time.time() + 0.200

        time.sleep_ms(10)

if __name__ == '__main__':
    main()
