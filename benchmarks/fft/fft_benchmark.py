
import math
import cmath
import time
import array
import gc

# Import the different implementations
from fft_python import FFTPreInplace

ulab = None
try:
#    import ulab
#    from ulab import numpy
    pass
except ImportError as e:
    print(e)

emlfft = None
try:
    import emlfft
    pass
except ImportError as e:
    print(e)


def make_two_sines(f1 = 2.0, f2 = 20.0, sr = 100, dur = 1.0):

    n = int(dur * sr)
    PI2 = 2 * math.pi

    a = array.array('f', ( math.sin(PI2*f1*t) + math.sin(PI2*f2*t) for t in range(n)) )
    return a


def run_one(real, imag, n, repeat=10):


    assert len(real) == n

    global ulab
    #ulab = True
    emlearn = True
    pyfft = True
    
    # Python
    """
    fft1 = FFTPreInplace(n)

    if pyfft:
        start = time.ticks_us()
        for i in range(repeat):
            fft1.compute(real, imag)
            #out = fft_optimized(data, seq)
        d = ((time.ticks_diff(time.ticks_us(), start)) / repeat) / 1000.0 # ms
        print('python', d)

    gc.collect()

    # ulab
    if ulab:
        arr = numpy.array(real)
        start = time.ticks_us()
        for i in range(repeat):
            out, _ = numpy.fft.fft(data)
        d = ((time.ticks_diff(time.ticks_us(), start)) / repeat) / 1000.0 # ms
        print('ulab', d)

    """

    # FIXME: this causes MicroPython to crash inside emlfft
    #gc.collect()

    # emlearn
    if emlearn:
        print(dir(emlfft))
        
        print("before new", gc.mem_free())
        time.sleep_ms(100)

        fft2 = emlfft.FFT(n)

        print("before fill")
        time.sleep_ms(100)

        #return
        
        emlfft.fill(fft2, n)

        print("filled")
        time.sleep_ms(100)

        start = time.ticks_us()
        for n in range(repeat):
            out = fft2.run(real, imag)
        d = ((time.ticks_diff(time.ticks_us(), start)) / repeat) / 1000.0 # ms
        print('emlearn', d)

    gc.collect()

def run_all():

    lengths = [
        128,
        #128,
        #256,
        #512,
        #1024,
    ]

    sines = make_two_sines(dur=1.5)
    print("fft-run-all22", lengths)

    for n in lengths:
        data = sines[0:n]
        imag = array.array('f', (0.0 for _ in range(n)))
        run_one(data, imag, n)
        gc.collect()


if __name__ == '__main__':
    run_all()
