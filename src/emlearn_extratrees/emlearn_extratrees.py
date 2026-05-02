# When used as external C module, the .py is the top-level import,
# and we need to merge the native module symbols at import time
# When used as dynamic native modules (.mpy), .py and native code is merged at build time
try:
    from emlearn_extratrees_c import *
except ImportError:
    pass


def new(n_features, n_classes, *,
        n_trees=10, max_depth=10, min_samples_leaf=1,
        n_thresholds=10, subsample_ratio=1.0, feature_subsample_ratio=1.0,
        max_nodes=1000, max_samples=1000,
        rng_seed=42, use_global_feature_range=False):
    """Create a new ExtraTrees model.

    Args:
        n_features: Number of input features (required)
        n_classes: Number of output classes (required)
        n_trees: Number of trees in the ensemble
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples at a leaf node
        n_thresholds: Random thresholds drawn per feature split
        subsample_ratio: Fraction of samples used per tree (0.0-1.0)
        feature_subsample_ratio: Fraction of features considered per split (0.0-1.0)
        max_nodes: Max pre-allocated nodes (memory)
        max_samples: Max pre-allocated samples (memory)
        rng_seed: Random number generator seed
        use_global_feature_range: Use global feature min/max instead of per-node
    """
    return make_new(
        n_features, n_classes,
        n_trees, max_depth, min_samples_leaf, n_thresholds,
        subsample_ratio, feature_subsample_ratio,
        max_nodes, max_samples,
        rng_seed, int(use_global_feature_range)
    )


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
