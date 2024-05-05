
import math
import cmath
import time
import array
import gc

# Import the different implementations
from fft_python import FFTPreInplace

ulab = None
try:
    import ulab
    from ulab import numpy
    pass
except ImportError as e:
    print(e)




def make_two_sines(f1 = 2.0, f2 = 20.0, sr = 100, dur = 1.0):
    np = numpy

    t = np.linspace(0, 1, num=int(dur*sr))
    sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

    return t, sig


def run_one(data, imag, n, repeat=10):

    emlfft = None
    try:
        import emlfft
        pass
    except ImportError as e:
        print(e)

    assert len(data) == n

    ulab = True
    emlearn = True
    pyfft = True
    
    re = array.array('f', data)
    im = array.array('f', imag)

    # Python
    fft1 = FFTPreInplace(n)

    if pyfft:
        start = time.ticks_us()
        for i in range(repeat):
            fft1.compute(re, im)
            #out = fft_optimized(data, seq)
        d = ((time.ticks_diff(time.ticks_us(), start)) / repeat) / 1000.0 # ms
        print('python', d)

    # ulab
    if ulab:    
        start = time.ticks_us()
        for i in range(repeat):
            out, _ = numpy.fft.fft(data)
        d = ((time.ticks_diff(time.ticks_us(), start)) / repeat) / 1000.0 # ms
        print('ulab', d)

    gc.collect()

    # emlearn
    if emlearn:
        print(dir(emlfft))
        
        print("before new", gc.mem_free())
        time.sleep_ms(100)

        fft2 = emlfft.FFT(n)

        print("before fill")
        time.sleep_ms(100)

        return
        
        emlfft.fill(fft2, n)

        print("filled")
        time.sleep_ms(100)

        start = time.ticks_us()
        for n in range(repeat):
            out = fft2.run(re, im)
        d = ((time.ticks_diff(time.ticks_us(), start)) / repeat) / 1000.0 # ms
        print('emlearn', d)

def run_all():

    lengths = [
        16,
        #128,
        #256,
        #512,
        #1024,
    ]

    sines = make_two_sines(dur=1.5)
    print("fft-run-all", lengths)

    for n in lengths:
        data = sines[0][0:n]
        imag = numpy.zeros(data.shape, dtype=data.dtype)
        run_one(data, imag, n)
        gc.collect()


if __name__ == '__main__':
    run_all()
