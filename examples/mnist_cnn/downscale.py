
import array
import time

@micropython.native
def average2d(inp, rowstride : int, x : int, y : int, size : int):
    acc : int = 0
    for r in range(y, y+size):
        for c in range(x, x+size):
            acc += inp[(r*rowstride)+c]

    avg = acc // (size*size)
    return avg

@micropython.native
def downscale(inp, out, in_size : int, out_size : int):

    # assumes square, single-channel (grayscale) images
    assert len(inp) == in_size*in_size
    assert len(out) == out_size*out_size
    assert (in_size % out_size) == 0, (in_size, out_size)
    factor : int = in_size // out_size

    for row in range(out_size):
        for col in range(out_size):
            o : int = (row * out_size) + col
            out[o] = average2d(inp, in_size, col*factor, row*factor, factor)


def test_downscale():

    import npyfile

    shape, data = npyfile.load('inp.npy')
    #print(data)

    npyfile.save('orig.npy', data, shape)

    insize = 96
    outsize = 32
    out = array.array('B', (0 for _ in range(outsize*outsize)))

    start = time.ticks_ms()
    downscale(data, out, insize, outsize)
    duration = time.ticks_diff(time.ticks_ms(), start)
    print('downscale', duration)

    npyfile.save('out.npy', out, (outsize, outsize))



if __name__ == '__main__':
    test_downscale()
