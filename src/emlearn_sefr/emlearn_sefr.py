
# When used as external C module, the .py is the top-level import,
# and we need to merge the native module symbols at import time
# When used as dynamic native modules (.mpy), .py and native code is merged at build time
try:
    from emlearn_sefr_c import *
except ImportError:
    pass


def train(model, X, y, lda=False):
    """Fit on a full dataset: X flattened float32, y float32 class indices.
    lda=True adds a second pass that combines the class scores with linear discriminant analysis"""
    model.reset()
    model.partial_fit(X, y)
    model.finalize()
    if lda:
        model.partial_fit_lda(X, y)
        model.finalize_lda()
