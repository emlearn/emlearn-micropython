
import gc
from machine import Pin, SPI

# mpremote mip install "github:peterhinch/micropython-nano-gui/drivers/st7789"
from drivers.st7789.st7789_4bit import *

board = 'T-WatchS3'
if board == 'M5StickPLUS2':

    # M5Stack M5StickC PLUS 2
    # https://docs.m5stack.com/en/core/M5StickC%20PLUS2
    #   ESP32 	    GPIO15 	    GPIO13 	    GPIO14 	GPIO12 	    GPIO5 	GPIO27
    #   TFT LCD 	TFT_MOSI 	TFT_CLK 	TFT_DC 	TFT_RST 	TFT_CS 	TFT_BL
    SSD = ST7789
    pdc = Pin(14, Pin.OUT, value=0)
    pcs = Pin(5, Pin.OUT, value=1)
    prst = Pin(12, Pin.OUT, value=1)
    pbl = Pin(27, Pin.OUT, value=1)

    gc.collect()  # Precaution before instantiating framebuf

    # Conservative low baudrate. Can go to 62.5MHz.
    spi = SPI(1, 30_000_000, sck=Pin(13), mosi=Pin(15))
    ssd = ST7789(spi, height=135, width=240, dc=pdc, cs=pcs, rst=prst, disp_mode=LANDSCAPE, display=TDISPLAY)

elif board == 'T-WatchS3':
    
    SSD = ST7789
    pdc = Pin(38, Pin.OUT, value=0)
    pcs = Pin(12, Pin.OUT, value=1)
    prst = None # Pin(2, Pin.OUT) # XXX: dummy Pin. There is no RST connected on this board
    pbl = Pin(45, Pin.OUT, value=1)

    gc.collect()  # Precaution before instantiating framebuf

    # FIXME: try to use https://github.com/antirez/t-watch-s3-micropython/blob/main/st7789_ext.py
    spi = SPI(1, 30_000_000, phase=0, polarity=1, sck=Pin(18), mosi=Pin(13))
    ssd = ST7789(spi, height=240, width=240, dc=pdc, cs=pcs, rst=prst, disp_mode=USD, display=TDISPLAY)
