
"""
Simple example for motion classification (Human Activity Detection et.c)
using emlearn-micropython

Takes tri-axial accelerometer data (int16) as fixed-length time windows,
computes overall energy features (p2p of magnitude),
as well as spectral features using FFT.

Then classifies the features using a tree-based classifier (Random Forest).
"""

import emlearn_fft
import emlearn_trees
import npyfile

import array

class AccelerometerClassifier():
    def __init__(self, window_length : int,
        max_trees=10, max_nodes=1000,
        dimensions=3, model_file=None, fft_start=0, fft_end=16):
        
        self.window_length = window_length
        self.dimensions = dimensions
        self.fft_start = fft_start
        self.fft_end = fft_end

        # Setup FFT
        fft_length = self.window_length
        self.fft = emlearn_fft.FFT(fft_length)
        emlearn_fft.fill(self.fft, fft_length)
        self.fft_real = array.array('f', (0 for _ in range(fft_length)))
        self.fft_imag = array.array('f', (0 for _ in range(fft_length)))

        # Setup tree-based classification model
        self.model = None
        self.probabilties = None
        if model_file is not None:
            self.model = emlearn_trees.new(max_trees, max_nodes, 10)
            with open(model_file) as f:
                emlearn_trees.load_model(self.model, f)
                classes = self.model.outputs()
                self.probabilities = array.array('f', (0 for _ in range(classes)))

    def preprocess(self, samples : array.array):
        assert len(samples) == (self.window_length*self.dimensions)
        samples_length = len(samples) // self.dimensions

        # Preprocess using FFT and simple statistics
        magnitude_min = float('inf')
        magnitude_max = -float('inf')
        for i in range(samples_length):
            x = samples[(i*3)+0] / 2**15
            y = samples[(i*3)+1] / 2**15
            z = samples[(i*3)+2] / 2**15
            magnitude = x*x + y*y + z*z
            if magnitude < magnitude_min:
                magnitude_min = magnitude
            if magnitude > magnitude_max:
                magnitude_max = magnitude
            self.fft_real[i] = magnitude
            self.fft_imag[i] = 0

        self.fft.run(self.fft_real, self.fft_imag)
        peak2peak = (magnitude_max - magnitude_min)
        # Normalize FFT by total energy
        fft_energy = sum(self.fft_real)
        if fft_energy > 1e-6:
            for i in range(len(self.fft_real)):
                self.fft_real[i] = 2**14 * (abs(self.fft_real[i]) / fft_energy)
        else:
            for i in range(len(self.fft_real)):
                self.fft_real[i] = 0.0

        # Pick relevant features
        fft_features = list(self.fft_real[self.fft_start:self.fft_end])
        features = [ 2**14 * peak2peak, 2**10 * fft_energy ] + fft_features
        return features

    def classify(self, features):
        self.model.predict(features, self.probabilities)
        return self.probabilities

def process_file(inp, out, model=None):

    window_length = 256
    pipeline = AccelerometerClassifier(window_length=window_length, model_file=model)
    fft_features = pipeline.fft_end - pipeline.fft_start    

    with npyfile.Reader(inp) as reader:

        # check input
        assert len(reader.shape) == 2
        assert reader.shape[1] == pipeline.dimensions
        assert reader.typecode == 'h' # int16

        # determine expected output
        chunksize = reader.shape[1]*pipeline.window_length
        n_windows = reader.shape[0]//pipeline.window_length
        out_dimensions = 2 + fft_features
        if model is not None:
            out_dimensions += pipeline.model.outputs()
        out_shape = (n_windows, out_dimensions)

        # process the data, using non-overlapped windows
        written = 0
        n_chunks = 0
        with npyfile.Writer(out, out_shape, 'f') as writer:

            for chunk in reader.read_data_chunks(chunksize):
                if len(chunk) < chunksize:
                    continue # last window chunk might be incomplete, skip it

                features = list(pipeline.preprocess(chunk))
                arr = array.array('h', [int(f) for f in features])
                if model:
                    probabilities = pipeline.classify(arr)
                else:
                    probabilities = []

                out = array.array('f', features + list(probabilities))
                assert len(out) == out_dimensions, (len(out), out_dimensions)
                writer.write_values(out)
                written += len(out)
                n_chunks += 1

if __name__ == '__main__':
    import sys
    assert len(sys.argv) >= 3
    model = sys.argv[3] if len(sys.argv) > 3 else None
    process_file(sys.argv[1], sys.argv[2], model)
