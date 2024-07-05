

"""
Color quantization
"""

import array
import time
import os

import emlkmeans


def apply_palette(img, quant, palette, rowstride):
    """Quantize an image to the specified palette"""

    # check palette
    channels = 3
    assert len(palette) % channels == 0, len(palette)
    n_palette = len(palette) // channels
    assert n_palette >= 2
    assert n_palette <= 256

    # check images
    assert len(img) % (channels*rowstride) == 0, len(img)
    assert len(quant) % (channels*rowstride) == 0, len(quant)

    rows = len(img) // (rowstride * 3)

    print('aa', rows, rowstride)

    for row in range(rows):
        for col in range(rowstride):
            i = 3 * (row*rowstride + col)
            rgb = array.array('B', img[i:i+3])

            # find closest value in palette
            #print('p', palette)
            #print('p', rgb)
            assert len(rgb) == 3
            palette_idx, distance = emlkmeans.euclidean_argmin(palette, rgb)

            #palette_idx, distance = 0, 0
    
            # copy the palette value
            p = palette_idx*3

            #print(palette_idx, distance, p, (r,g,b), tuple(palette[p:p+3]))

            # TODO: only output palette index values
            quant[i+0] = palette[p+0]
            quant[i+1] = palette[p+1]
            quant[i+2] = palette[p+2]

    pass

def make_image(width, height, channels=3, typecode='B', value=0):
    """Utility to create image stored as 1d buffer/array"""
    
    elements = width * height * channels
    img = array.array(typecode, (value for _ in range(elements)))

    return img

def sample_pixels(img, samples, n, channels=3):
    import random

    assert (len(samples) // channels) >= n, len(samples)
    elements = len(img) // channels

    for i in range(n):
        e = random.randint(0, elements)
        samples[i*channels:(i+1)*channels] = img[e*channels:(e+1)*channels]



def quantize_path(inp, outp, palette, n_samples=100):
    # https://github.com/jacklinquan/micropython-microbmp
    from microbmp import MicroBMP

    loaded = MicroBMP().load(inp)
    res = (loaded.DIB_w, loaded.DIB_h)
    print('loaded image of dimensions', res)

    # TODO: use 8 or 4 bit palette instead of full color
    out = MicroBMP(res[0], res[1], 24)

    # Sample some pixels
    # This reduces the computational requirements of ths clustering step a lot
    samples = make_image(1, n_samples)

    start = time.ticks_us()
    sample_pixels(loaded.parray, samples, n=n_samples)
    dur = (time.ticks_diff(time.ticks_us(), start) / 1000.0)
    print('sample duration (ms)', dur)

    # Learn a palette
    start = time.ticks_us()
    emlkmeans.cluster(samples, palette, max_iter=20)
    dur = (time.ticks_diff(time.ticks_us(), start) / 1000.0)
    print('cluster duration (ms)', dur)

    # Show selected palette
    for i in range(len(palette)//3):
        print(palette[(i*3):(i*3)+3])

    # Apply palette
    start = time.ticks_us()
    apply_palette(loaded.parray, out.parray, palette, rowstride=res[1])

    dur = (time.ticks_diff(time.ticks_us(), start) / 1000.0)
    print('apply palette duration (ms)', dur)

    # Save output
    out.save(outp)


# RAL Standard Color Table
# https://gist.github.com/nichtich/4a8a110a4d3f8f0e2baa593f26d50d9d
# RAL classic color table
# https://gist.github.com/lunohodov/1995178
# EGA 64
# https://en.wikipedia.org/wiki/List_of_8-bit_computer_hardware_graphics
# https://en.wikipedia.org/wiki/List_of_16-bit_computer_color_palettes
PALETTE_EGA16_HEX = [
    '#ffffff',
    '#000000',
    '#0000aa',
    '#00aa00',
    '#00aaaa',
    '#aa0000',
    '#aa00aa',
    '#aa5500',
    '#aaaaaa',
    '#555555',
    '#5555ff',
    '#55ff55',
    '#55ffff',
    '#ff5555',
    '#ff55ff',
    '#ffff55',
]

def hex_to_rgb8(s : str) -> tuple:
    assert s[0] == '#'

    r = int(s[1:3], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return r, g, b


def main():

    inp = 'IMG_20240626_175314_MP_cifm.bmp'
    out = 'quant.bmp'

    # Load a fixed palette
    hh = PALETTE_EGA16_HEX
    palette = make_image(1, len(hh))
    for i, h in enumerate(hh):
        rgb = hex_to_rgb8(h)
        c = array.array('B', rgb)
        #print(i, rgb, c)
        palette[(i*3):(i*3)+3] = c

    quantize_path(inp, out, palette)

if __name__ == '__main__':
    main()


