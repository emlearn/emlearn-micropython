
import math
import struct
import array
from collections import deque

import emliir
from emlearn_arrayutils import linear_map
#import emlearn_iir_q15
#from iir_python import IIRFilter

class IIRFilterEmlearn:

    def __init__(self, coefficients):
        c = array.array('f', coefficients)
        self.iir = emliir.new(c)
    def process(self, samples):
        self.iir.run(samples)

class IIRFilterEmlearnFixed:

    def __init__(self, coefficients):
        c = emlearn_iir_q15.convert_coefficients(coefficients)
        self.iir = emlearn_iir_q15.new(c)
    def process(self, samples):
        self.iir.run(samples)


# A method for computing A weighting filters etc for any sample-rate
# https://www.dsprelated.com/thread/10546/a-weighting-filter

def assert_array_typecode(arr, typecode):
    actual_typecode = str(arr)[7:8]
    assert actual_typecode == typecode, (actual_typecode, typecode)

a_filter_16k = [
    1.0383002230320646, 0.0, 0.0, 1.0, -0.016647242439959593, 6.928267021369795e-05,
    1.0, -2.0, 1.0, 1.0, -1.7070508390293027, 0.7174637059318595,
    1.0, -2.0, 1.0, 1.0, -1.9838868447331497, 0.9839517531763131
]

@micropython.native
def rms_micropython_native(arr):
    acc = 0.0
    for i in range(len(arr)):
        v = arr[i]
        p = (v * v)
        acc += p
    mean = acc / len(arr)
    out = math.sqrt(mean)
    return out

# Using a limited-precision aware approach based on Cumulative Moving Average
# https://www.codeproject.com/Articles/807195/Precise-and-safe-calculation-method-for-the-averag
@micropython.viper
def rms_micropython_viper(arr) -> object:
    buf = ptr16(arr) # XXX: input MUST BE h/uint16 array
    l = int(len(arr))
    cumulated_average : int = 0
    cumulated_remainder : int = 0
    addendum : int = 0
    n_values : int = 0
    for i in range(l):
        v = int(buf[i])
        value = (v * v) # square it
        n_values += 1
        addendum = value - cumulated_average + cumulated_remainder
        cumulated_average += addendum // n_values
        cumulated_remainder = addendum % n_values

    # sqrt it
    out = math.sqrt(cumulated_average)
    return out

@micropython.native
def time_integrate_native(arr, initial, time_constant, samplerate):
    acc = initial
    dt = 1.0/samplerate
    a = dt / (time_constant + dt)
    #print('a', a)

    for i in range(len(arr)):
        v = arr[i]
        p = (v * v) # power is amplitude squared
        # exponential time weighting aka 1st order low-pass filter
        #acc = (a*p) + ((1-a)*acc) 
        acc = acc + a*(p - acc) # exponential time weighting aka 1st order low-pass filter
        #acc += p

    return acc

# Use C module for data conversion
# @micropython.native with a for loop is too slow
def int16_to_float(inp, out):
    return linear_map(inp, out, -2**15, 2**15, -1.0, 1.0)

def float_to_int16(inp, out):
    return linear_map(inp, out, -1.0, 1.0, -2**15, 2**15)


class Summarizer():
    """Compute common acoustical summarizations of soundlevels"""

    def __init__(self, maxlen):
        self._capacity = maxlen
        self._deque = deque([], maxlen, 1)
    
    def reset(self):
        while len(self._deque):
            self._deque.popleft()

    def full(self):
        full = len(self._deque) == self._capacity
        return full

    def push(self, value):
        # if full, drop oldest value
        if len(self._deque) >= self._capacity:
            print('summarizer-queue-overflow')
            self._deque.popleft()
    
        self._deque.append(value)
    
    def compute_leq(self):
        # NOTE: assumes that the values in the deque are decibel values
        avg = sum((pow(10, db/10.0) for db in self._deque)) / len(self._deque)
        leq = 10*math.log10(avg)
        return leq

    def compute_minmax(self):
        mn = min(self._deque)
        mx = max(self._deque)
        return mn, mx

    def compute_percentiles(self, percentiles : list[float]) -> list[float]:
        values = sorted(self._deque)
        out = []
        for p in percentiles:
            # find closest value.
            # XXX: no interpolation
            idx = round(self._capacity * (p/100.0))
            #print(p, idx, idx/self._capacity, self._capacity, len(self._deque))
            out.append(values[idx])

        return out

    def compute_all(self, levels=(10, 50, 90)) -> dict[str, float]:

        leq = self.compute_leq()
        lmin, lmax = self.compute_minmax()
        metrics = {
            'Lmin': lmin,
            'Lmax': lmax,
            'Leq': leq,
        }
        ln_values = self.compute_percentiles([100-ln for ln in levels])
        for ln, value in zip(levels, ln_values):
            metrics[f'L{ln}'] = value

        return metrics


class SoundlevelMeter():

    def __init__(self, buffer_size,
        samplerate,
        mic_sensitivity,
        time_integration=0.125,
        frequency_weighting='A',
        summary_interval=60.0,
        summary_capacity=2,
        ):

        buffer_duration = samplerate / float(buffer_size)

        self._buffer_size = buffer_size
        self._sensitivity_dbfs = mic_sensitivity

        buffer_duration = buffer_size / samplerate
        assert buffer_duration <= 0.125

        self._power_integrated_fast = 0.0
        self._samplerate = samplerate
        self._time_integration = time_integration

        if not frequency_weighting:
            self.frequency_filter = None
        elif frequency_weighting == 'A':
            #self.frequency_filter = IIRFilter(a_filter_16k)
            self.frequency_filter = IIRFilterEmlearn(a_filter_16k)
            #self.frequency_filter = IIRFilterEmlearnFixed(a_filter_16k)

            self.float_array = array.array('f', (0 for _ in range(buffer_size)))
        else:
            raise ValueError('Unsupported frequency_weighting')

        self._summary_interval = summary_interval
        self._summary_capacity = summary_capacity
        per_summary_interval = int((1/buffer_duration)*summary_interval)
        self._summarizer = Summarizer(per_summary_interval)
        self._summary_queue = deque([], self._summary_capacity)

        self._last_value = None

    def last_value(self):
        return self._last_value

    def compute_level(self, samples):
        assert len(self.float_array) == self._buffer_size
        assert len(samples) == self._buffer_size
        assert_array_typecode(samples, 'h')

        # Apply frequency weighting
        if self.frequency_filter:
            int16_to_float(samples, self.float_array)
            self.frequency_filter.process(self.float_array)
            float_to_int16(self.float_array, samples)

        spl_max = 94 - self._sensitivity_dbfs
        ref = 2**15

        # no integration - "linear"
        if self._time_integration is None:
            rms = rms_micropython_native(samples)
            # FIXME: causes math domain error
            #rms = rms_micropython_viper(samples)
        else:
            p = time_integrate_native(samples,
                self._power_integrated_fast,
                self._time_integration,
                self._samplerate,
            )
            self._power_integrated_fast = p
            rms = math.sqrt(p)

        level = 20*math.log10(rms/ref)
        level += (spl_max)

        self._last_value = level
        return level

    def process(self, samples):
        # compute soundlevel for this instant
        level = self.compute_level(samples)

        if self._summary_interval > 0.0:
            self._summarizer.push(level)

            # update summarized metrics
            if self._summarizer.full():
                metrics = self._summarizer.compute_all()
                if len(self._summary_queue) >= self._summary_capacity:
                    self._summary_queue.popleft() # drop oldest
                    print('summary-queue-overflow')
                self._summary_queue.append(metrics)
                self._summarizer.reset()

        return level


