
import math
import array
import time

from iir_python import IIRFilter

ulab = None
try:
    import ulab
    from ulab import numpy
    from ulab import scipy
except ImportError as e:
    print(e)
    pass

emlearn_iir = None
try:
    import emlearn_iir
except ImportError as e:
    print(e)
    pass


def make_two_sines(f1 = 2.0, f2 = 20.0, sr = 100, dur = 1.0):
    n = int(dur * sr)
    PI2 = 2 * math.pi

    a = array.array('f', ( math.sin(PI2*f1*t) + math.sin(PI2*f2*t) for t in range(n)) )
    return a


def main():

    cutoff = 10.0
    sr = 100
    # XXX: note, adding the 2-order filters like this does not yield a proper Butterworth response
    # but this is more than good enough for benchmarking
    sos = [
        butter2_lowpass(cutoff, sr),
        butter2_lowpass(cutoff, sr),
        butter2_lowpass(cutoff, sr),
        butter2_lowpass(cutoff, sr),
    ]
    coeff = []
    for s in sos:
        coeff += s
    coeff = array.array('f', coeff)
    print('cc', len(coeff))

    repeats = 1000

    a = make_two_sines(dur=4.0, sr=100)

    # Pure Python
    iir = IIRFilter(sos)
    start = time.ticks_us()
    for r in range(repeats):
        iir.process(a)
        t = (time.ticks_diff(time.ticks_us(), start) / repeats ) / 1000.0 # ms
    print('python', t)

    # ulab
    if ulab:
        inp = numpy.array(a) # convert to ulab array
        start = time.ticks_us()
        for r in range(repeats):
            scipy.signal.sosfilt(sos, inp)
        t = (time.ticks_diff(time.ticks_us(), start) / repeats ) / 1000.0 # ms
        print('ulab', t)

    # emlearn
    if emlearn_iir:
        start = time.ticks_us()
        iir = emlearn_iir.new(coeff)
        for r in range(repeats):
            iir.run(a)
        t = (time.ticks_diff(time.ticks_us(), start) / repeats ) / 1000.0 # ms
        print('emlearn', t)


if __name__ == '__main__':
    main()
