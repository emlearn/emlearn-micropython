
import gc
from machine import Pin, SPI

# M5Stack M5StickC PLUS 2
# https://docs.m5stack.com/en/core/M5StickC%20PLUS2
from drivers.st7789.st7789_4bit import *
#   ESP32 	    GPIO15 	    GPIO13 	    GPIO14 	GPIO12 	    GPIO5 	GPIO27
#   TFT LCD 	TFT_MOSI 	TFT_CLK 	TFT_DC 	TFT_RST 	TFT_CS 	TFT_BL

pdc = Pin(14, Pin.OUT, value=0)
pcs = Pin(5, Pin.OUT, value=1)
prst = Pin(12, Pin.OUT, value=1)
pbl = Pin(27, Pin.OUT, value=1)

gc.collect()  # Precaution before instantiating framebuf

SSD = ST7789

# Conservative low baudrate. Can go to 62.5MHz.
spi = SPI(1, 30_000_000, sck=Pin(13), mosi=Pin(15))
ssd = ST7789(spi, height=135, width=240, dc=pdc, cs=pcs, rst=prst, disp_mode=LANDSCAPE, display=TDISPLAY)

