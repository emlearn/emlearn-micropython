
"""MicroPython code for doing multi-output regression with emlearn_trees
"""

import os

import emlearn_trees
import array

class MultiRegressor():
    """Convenience wrapper for a collection of tree-based regression models"""

    def __init__(self, max_trees=10, max_nodes=1000, max_leaves=1000):
        self.models = []

        self.max_trees = max_trees
        self.max_nodes = max_nodes
        self.max_leaves = max_leaves        

        # temporary buffer for invididual model output
        self._output = array.array('f', [0.0])

    def load(self, path):
        """Load a directory of model files"""

        for filename in sorted(os.listdir(path)):
            if not filename.endswith('.csv'):
                print('Warning: Ignoring unknown file in model directory', filename)
                continue

            model_path = path + '/' + filename

            # TODO: support reading neccesary capacity from file
            model = emlearn_trees.new(self.max_trees, self.max_nodes, self.max_leaves)

            with open(model_path, 'r') as f:
                emlearn_trees.load_model(model, f)

            self.models.append(model)

    def predict(self, features : array.array, outputs : array.array):
        assert len(self.models), 'no models'

        for i, model in enumerate(self.models):
            model.predict(features, self._output)
            outputs[i] = self._output[0]

def main():
    
    # FIXME: read paths from sys.argv
    model = MultiRegressor(max_nodes=10000)
    model.load('models')

    outputs = array.array('f', [0.0 for _ in range(len(model.models))])

    import npyfile
    (n_samples, n_features), data = npyfile.load('input.npy')

    # TODO: write output to a file
    for row in range(n_samples):
        offset = row*n_features
        f = data[offset:offset+n_features]
        model.predict(f, outputs)
        print(f, outputs)
    

if __name__ == '__main__':
    main()
