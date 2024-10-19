

import sys
import struct
import array

import npyfile

#from emlearn_melspectrogram import Processor

from miniwav import read_wav, get_wav_samples

import emlearn_fft
from emlearn_arrayutils import linear_map

def int16_to_float(inp, out):
    #print('int16tofloat', array_typecode(inp), array_typecode(out))
    return linear_map(inp, out, -2**15, 2**15, -1.0, 1.0)

@micropython.native
def spectrum_average_bins(spec, out):
    binsize = len(spec) // len(out)

    for bin in range(len(out)):
        s = 0.0
        for i in range(bin*binsize, (bin+1)*binsize):
            s += spec[i]        
        out[bin] = s / binsize
    
def array_typecode(arr):
    typecode = str(arr)[7:8]
    return typecode

class Processor():

    def __init__(self, fft_length, hop_length, n_mels, fmin, fmax):
        pass
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.fft = emlearn_fft.FFT(fft_length)
        emlearn_fft.fill(self.fft, fft_length)
        self.fft_length = fft_length
        self.fft_imag = array.array('f', (0.0 for _ in range(fft_length)))

        self.buffer = array.array('h', (0 for _ in range(fft_length)))
        self.buffer_valid = 0

        self.fft_buffer = array.array('f', (0.0 for _ in range(fft_length)))

    def process(self, chunk, mels):
        assert len(chunk) == self.hop_length
        assert len(mels) == self.n_mels

        # shift new data into our buffer
        available = self.fft_length - self.buffer_valid
        shift = max(len(chunk) - available, 0) 

        #print('shift', shift)
        self.buffer[0:self.buffer_valid] = self.buffer[shift:shift+self.buffer_valid]
        self.buffer_valid += (len(chunk) - shift)

        insert = self.buffer_valid - len(chunk)
        self.buffer[insert:] = chunk

        if self.buffer_valid < self.fft_length:
            # not enough samples to get valid output yet
            return False

        #print(chunk)
        #print(self.buffer)
        int16_to_float(self.buffer, self.fft_buffer)

        # TODO: apply windowing (Hann)

        # perform FFT
        for i in range(len(self.fft_imag)):
            self.fft_imag[i] = 0.0
        self.fft.run(self.fft_buffer, self.fft_imag)

        #if self.fft_len

        for i in range(len(mels)):
            mels[i] = self.fft_buffer[i]

        # Downscale
        # FIXME: use mel filterbank instead
        #spectrum_average_bins(self.fft_buffer, mels)

        # TODO: decibel/log scale the results

        return True

def process_file(audiofile,
        hop_length = 512,
        fft_length = 1024,
        sr = 16000,
        n_mels = 32,
        fmin = 50,
        fmax = None,
    ):
    """Compute spectrogram for a .wav file"""

    if fmax is None:
        fmax = sr // 2

    mels = array.array('f', (0 for _ in range(n_mels)))
    processor = Processor(fft_length, hop_length, n_mels, fmin, fmax)

    samples_read = 0

    t = 0.0 # seconds
    for chunk in read_wav(audiofile, sr, frames=hop_length):
        
        samples_read += len(chunk)
        if len(chunk) < hop_length:
            # EOF
            print('eof', samples_read, samples_read/sr)
            return

        # NOTE: may want to do spectral subtraction? streaming?
        # And scale normalization (probably of an entire window of N frames)
        processor.process(chunk, mels)

        dt = len(chunk)/float(sr)
        t += dt

        yield t, mels




def main():


    # micropython examples/melspectrogram/melspectrogram_file.py  test_burst.wav out.npy 16000 512 1024 32 0 8000

    if len(sys.argv) != 9:
        raise ValueError('Usage: micropython melspectrogram_file.py AUDIO.wav SPEC.npy samplerate hop fft nmels fmin fmax')

    # Get parameters
    audio_path = sys.argv[1]
    out_path = sys.argv[2]
    params = sys.argv[3:]
    sr, hop_length, fft_length, n_mels, fmin, fmax = (int(p) for p in params)

    n_mels = fft_length # XXX: hack

    samples = get_wav_samples(audio_path)

    spec_length = samples // hop_length

    out_shape = (spec_length, n_mels)
    out_typecode = 'f'

    print('ss', samples, out_shape)

    with open(audio_path, 'rb') as audiofile:
        with npyfile.Writer(out_path, out_shape, out_typecode) as out:

            for t, mels in process_file(audiofile, n_mels=n_mels):
                print(t, len(mels))
                out.write_values(mels)


if __name__ == '__main__':
    main()
    
