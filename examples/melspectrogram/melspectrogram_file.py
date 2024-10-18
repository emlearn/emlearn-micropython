

import sys
import struct
import array

import npyfile

#from emlearn_melspectrogram import Processor

from miniwav import read_wav, get_wav_samples

class Processor():

    def __init__(self, fft_length, hop_length, n_mels, fmin, fmax):
        pass
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(self, chunk, mels):
        assert len(chunk) == self.hop_length
        assert len(mels) == self.n_mels


def process_file(audiofile,
        hop_length = 512,
        fft_length = 1024,
        sr = 16000,
        n_mels = 32,
        fmin = 50,
        fmax = None,
    ):
    """Compute soundlevels for a .wav file"""

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

    # TODO: write output to a .npy file using npyfile

    # Get parameters
    audio_path = sys.argv[1]
    out_path = sys.argv[2]
    params = sys.argv[3:]
    sr, hop_length, fft_length, n_mels, fmin, fmax = (int(p) for p in params)

    samples = get_wav_samples(audio_path)

    print('ss', samples)

    spec_length = samples // hop_length

    out_shape = (n_mels, spec_length)
    out_typecode = 'f'

    with open(audio_path, 'rb') as audiofile:
        with npyfile.Writer(out_path, out_shape, out_typecode) as out:

            for t, mels in process_file(audiofile):
                #print(t, len(mels))
                out.write_values(mels)


if __name__ == '__main__':
    main()
    
