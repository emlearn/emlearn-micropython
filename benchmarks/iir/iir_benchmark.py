
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

emliir = None
try:
    import emliir
except ImportError as e:
    print(e)
    pass

# https://stackoverflow.com/a/20932062
def butter2_lowpass(f, sr):
    ff = f / sr
    ita = 1.0/ math.tan(math.pi*ff)
    q = math.sqrt(2.0)
    b0 = 1.0 / (1.0 + q*ita + ita*ita)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita*ita - 1.0) * b0
    a2 = -(1.0 - q*ita + ita*ita) * b0

    # Return in biquad / Second Order Stage format
    # to be compatible with scipy, a1 and a2 needed to be flipped??
    sos = [ b0, b1, b2, 1.0, -a1, -a2 ]

    return sos

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
    if emliir:
        start = time.ticks_us()
        iir = emliir.new(coeff)
        for r in range(repeats):
            iir.run(a)
        t = (time.ticks_diff(time.ticks_us(), start) / repeats ) / 1000.0 # ms
        print('emlearn', t)


if __name__ == '__main__':
    main()
