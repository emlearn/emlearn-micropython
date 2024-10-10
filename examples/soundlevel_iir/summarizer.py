
import math
from collections import deque

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
