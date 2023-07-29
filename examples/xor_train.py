# python/host code

import emlearn
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer

# Generate simple dataset
def make_xor(lower=0.0, upper=1.0, threshold=0.5, samples=100, seed=42):
    rng = numpy.random.RandomState(seed)
    X = rng.uniform(lower, upper, size=(samples, 2))
    y = numpy.logical_xor(X[:, 0] > threshold, X[:, 1] > threshold)
    return X, y

X, y = make_xor()

# Train a model
estimator = RandomForestClassifier(n_estimators=3, max_depth=3, max_features=2, random_state=1)
estimator.fit(X, y)
score = get_scorer('f1')(estimator, X, y)
assert score > 0.90, score # verify that we learned the function

# Convert model using emlearn
cmodel = emlearn.convert(estimator, method='inline')

# Save model
def save_csv(trees):
    """
    Code to export emlearn model as CSV format, loadable by emlearn-micropython
    """

    nodes, roots = trees.forest_
    nodes = nodes.copy()
    lines = []
    for r in roots:
        lines.append(f'r,{r}')
    for n in nodes:
        lines.append(f'n,{n[0]},{n[1]},{n[2]},{n[3]}')
    code = '\r\n'.join(lines) 
    return code

path = 'xor_model.csv'
exported = save_csv(cmodel)
with open(path, 'w') as f: 
    f.write(exported)
print('Wrote model to', path)
