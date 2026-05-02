# When used as external C module, the .py is the top-level import,
# and we need to merge the native module symbols at import time
# When used as dynamic native modules (.mpy), .py and native code is merged at build time
try:
    from emlearn_extratrees_c import *
except ImportError:
    pass


def train_steps(model, X, y):
    """Generator that trains one node at a time, yielding after each step.

    Yields the number of trees trained so far after each step.
    Returns when training is complete.

    Usage:
        for trees_done in model.train_steps(X, y):
            # do other work here (handle sensors, UI, etc.)
            pass
    """
    model.train_init(X, y)
    while True:
        done = model.train_step()
        if done:
            break
        yield model.get_n_trees_trained()
