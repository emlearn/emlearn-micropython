

import time
import array

# Using https://github.com/cnadler86/micropython-camera-API for ESP32 with OV2640 camera
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

import downscale

def print_2d_buffer(arr, rowstride):

    rows = len(arr) // rowstride
    columns = rowstride

    for r in range(rows):
        for c in range(columns):
            v = arr[(r*rowstride)+c]
            print('{v:03d}'.format(v=v), end='')

        gc.collect()
        print('\n')




def setup_camera():

    # Pin configuration for LilyGO TTGO LilyGO TTGO T-Camera Mic ESP32
    DATA_PINS = [
        34,
        13,
        14,
        35,
        39,
        12,
        15,
        36,
    ]

    # FIXME: set up camera power using the PMU
    # https://github.com/Xinyuan-LilyGO/LilyGo-Cam-ESP32S3/blob/master/examples/MinimalPowersExample/MinimalPowersExample.ino
    # https://github.com/CDarius/CircuitPython_AXP2101/blob/main/axp2101.py
    # https://github.com/Xinyuan-LilyGO/LilyGo-Cam-ESP32S3/tree/master

    # T-Camera S3
    DATA_PINS = [
        14,
        47,
        48,
        21,
        13,
        11,
        10,
        9,
    ]

    camera = Camera(
        data_pins=DATA_PINS,
        vsync_pin=8,
        href_pin=18,
        sda_pin=5, # XXX: is the I2C correct?
        scl_pin=4,
        pclk_pin=12,
        xclk_pin=38,
        xclk_freq=20000000,
        powerdown_pin=-1,
        reset_pin=39,
        pixel_format=PixelFormat.GRAYSCALE,
        frame_size=FrameSize.R96X96,
        jpeg_quality=15,
        fb_count=1,
        grab_mode=GrabMode.WHEN_EMPTY
    )
    return camera

def compute_mean(arr, rowstride):
    agg = 0.0

    rows = len(arr) // rowstride
    columns = rowstride

    for r in range(rows):
        for c in range(columns):
            v = arr[(r*rowstride)+c]
            agg += float(v)
    
    mean = agg / len(arr)

    return mean


def save_image(path, arr, width, height):

    # https://github.com/jacklinquan/micropython-microbmp
    # wget https://raw.githubusercontent.com/jacklinquan/micropython-microbmp/refs/heads/main/microbmp.py        
    from microbmp import MicroBMP

    if len(arr) != (width*height):
        raise ValueError("Unexpected size")

    out = MicroBMP(width, height, 8)
    for r in range(height):
        for c in range(width):
            i = (r*width)+c
            out.parray[i] = arr[i]

    # Save output
    out.save(path)

def main():

    camera = setup_camera()
    width = 96
    height = 96
    print('after constructor')

    image_no = 0

    process_size = 32
    scaled = array.array('B', (0 for _ in range(process_size*process_size)))

    while True:

        capture_start = time.ticks_ms()
        print('start capture')

        # Capture image
        img = camera.capture()

        print(len(img))
        buf = bytes(img)
        print(len(buf))

        capture_duration = time.ticks_diff(time.ticks_ms(), capture_start)

        print('end capture', capture_duration)

        mean = compute_mean(buf, width)
        print('mean', mean)

        path = f'img{image_no}.bmp'
        #save_image(path, buf, width, height)
        #print_2d_buffer(buf, 96)
        #print('Saved', path)

        # Downscale
        downscale_start = time.ticks_ms()
        downscale.downscale(buf, scaled, width, process_size)
        downscale_duration = time.ticks_diff(time.ticks_ms(), downscale_start)
        print('downscale', downscale_duration)
        # TODO: run CNN inference

        image_no += 1
        time.sleep(5.0)

main()

