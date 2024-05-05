
"""
FFT implementation in pure Python for MicroPython

Trying to use as many tricks as possible (while being floating point) to make it fast:

- Cooley-Tukey Radix-2 decimation-in-time in-place algorithm
- Precomputed bit-reversal table
- Precomputed trigonometry tables
- @micropython.native  / @micropython.viper

NOTE: still very slow compared to ulab or emlearn FFT.
Potentially 100-1000x for 512+ lengths.
"""

def reverse_bits(index, length):

    # Compute levels = floor(log2(n))
    levels = 0
    temp = length
    while temp > 1:
        temp = (temp >> 1)
        levels +=1

    result = 0
    x = index
    for i in range(levels):
        result = (result << 1) | (x & 1)
        x = (x >> 1)

    return result

class FFTPreInplace:

    def __init__(self, length):

        self.length = length
        self.bit_reverse_table = array.array('h', (reverse_bits(i, length) for i in range(length)))
        self.cos_table = array.array('f', (math.cos(2.0*math.pi*i/length) for i in range(length)) )
        self.sin_table = array.array('f', (math.sin(2.0*math.pi*i/length) for i in range(length)) )

    @micropython.native
    def compute(self, real, imag):
        # check inputs
        assert len(real) == self.length
        assert len(imag) == self.length

	    # Bit-reversed addressing permutation
        # does not compile with viper
        for ii in range(self.length):
            i = int(ii)
            j = self.bit_reverse_table[i]
            if j > i:
                temp : object = real[i]
                real[i] = real[j]
                real[j] = temp
                temp = imag[i]
                imag[i] = imag[j]
                imag[j] = temp

        self._compute(real, imag)

    @micropython.viper
    def _compute(self, real, imag):

        cos = self.cos_table
        sin = self.sin_table
        n : int = int(self.length)

    	## Cooley-Tukey in-place decimation-in-time radix-2 FFT
        size : int = 2
        while size <= n:
            halfsize : int = size // 2
            tablestep : int = n // size

            i = 0
            while i < n:
                k : int = 0
                j = i
                while j < i+halfsize:
                    l : int = j + halfsize

                    #tpre =  real[l] * cos[k] + imag[l] * sin[k]
                    #tpim = -real[l] * sin[k] + imag[l] * cos[k]
                    # splitting these gives 25% speedup
                    c = cos[k]
                    s = sin[k]
                    #iii = 2.0*math.pi*k/n
                    #c = math.cos(iii)
                    #s = math.sin(iii)
                    r = real[l]
                    im = imag[l]
                    tpre =  r * c + im * s
                    tpim = -r * s + im * c
                    real[l] = real[j] - tpre
                    imag[l] = imag[j] - tpim
                    real[j] += tpre
                    imag[j] += tpim
                    k += tablestep
                    j += 1

                i += size

            size = size * 2
