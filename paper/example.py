
import emlearn_fft
import emlearn_trees
import npyfile

import array

class AccelerometerClassifier():
    def __init__(self, window_length : int, classes : int,
        max_trees=10, max_nodes=1000,
        dimensions=3, model_file=None):
        
        self.window_length = window_length
        self.dimensions = dimensions

        # Setup FFT
        fft_length = self.window_length
        self.fft = emlearn_fft.FFT(fft_length)
        emlearn_fft.fill(self.fft, fft_length)
        self.fft_real = array.array('f', (0 for _ in range(fft_length)))
        self.fft_imag = array.array('f', (0 for _ in range(fft_length)))

        # Setup tree-based classification model
        self.model = None
        self.probabilities = array.array('f', (0 for _ in range(classes)))
        if model_file is not None:
            self.model = emlearn_trees.new(10, 1000, 10)
            with open('model.trees.csv') as f:
                emlearn_trees.load_model(self.model, f)

    def preprocess(self, samples : array.array):
        assert len(samples) == (self.window_length*self.dimensions)
        samples_length = len(samples) // self.dimensions

        # Preprocess using FFT and simple statistics
        magnitude_min = float('inf')
        magnitude_max = -float('inf')
        for i in range(samples_length):
            x = samples[(i*3)+0]
            y = samples[(i*3)+1]
            z = samples[(i*3)+2]
            magnitude = x*x + y*y + z*z
            if magnitude < magnitude_min:
                magnitude_min = magnitude
            if magnitude > magnitude_max:
                magnitude_max = magnitude
            self.fft_real[i] = magnitude
            self.fft_imag[i] = 0

        self.fft.run(self.fft_real, self.fft_imag)

        peak2peak = magnitude_max - magnitude_min

        # Pick relevant features
        features = ( peak2peak, self.fft_real[2], )
        return features

    def classify(self, features):
        self.model.predict(features, self.probabilities)
        return self.probabilities

def process_file(inp, out, model=None):

    pipeline = AccelerometerClassifier(window_length=32, classes=5, model_file=model)
    
    with npyfile.Reader(inp) as reader:

        # check input
        assert len(reader.shape) == 2
        assert reader.shape[1] == pipeline.dimensions
        assert reader.typecode == 'h'

        # determine expected output
        chunksize = reader.shape[1]*pipeline.window_length
        n_windows = reader.shape[0]//pipeline.window_length
        out_dimensions = 2
        if model is not None:
            out_dimensions += pipeline.n_classes
        out_shape = (n_windows, out_dimensions)

        # process the data
        written = 0
        n_chunks = 0
        with npyfile.Writer(out, out_shape, 'f') as writer:

            for chunk in reader.read_data_chunks(chunksize):
                if len(chunk) < chunksize:
                    continue # last window chunk might be incomplete, skip it

                features = list(pipeline.preprocess(chunk))
                if model:
                    probabilities = pipeline.classify(features)
                else:
                    probabilities = []

                out = array.array('f', features + probabilities)
                assert len(out) == out_dimensions, (len(out), out_dimensions)
                writer.write_values(out)
                written += len(out)
                n_chunks += 1
        print(out_shape, written, n_chunks, written/n_chunks)

if __name__ == '__main__':
    import sys
    assert len(sys.argv) >= 3
    model = sys.argv[3] if len(sys.argv) > 3 else None
    process_file(sys.argv[1], sys.argv[2], model)
