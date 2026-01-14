
# When used as external C module, the .py is the top-level import,
# and we need to merge the native module symbols at import time
# When used as dynamic native modules (.mpy), .py and native code is merged at build time
try:
    from emlearn_trees_c import *
except ImportError as e:
    pass

def load_model(builder, f):

    leaves_found = 0
    n_classes = None
    n_features = None
    leaf_bits = 0 # default if not specified

    for line in f:
        line = line.rstrip('\r')
        line = line.rstrip('\n')
        tok = line.split(',')
        kind = tok[0]

        missing = leaf_bits is None or n_features is None or n_classes is None

        if kind == 'r':
            assert not missing, 'missing metadata before roots'
            root = int(tok[1])
            builder.addroot(root)
        elif kind == 'n':
            assert not missing, 'missing metadata before nodes'
            feature = int(tok[1])
            value = int(float(tok[2]))
            left = int(tok[3])
            right = int(tok[4])
            builder.addnode(left, right, feature, value)
        elif kind == 'l':
            assert not missing, 'missing metadata before leaves'
            assert len(tok) == 2, len(tok)
            if leaf_bits == 32:
                leaf = float(tok[1])
            else:
                leaf = int(tok[1])
            builder.addleaf(leaf)
            leaves_found += 1
        # metadata
        elif kind == 'f':
            n_features = int(tok[1])
        elif kind == 'c':
            n_classes = int(tok[1])
        elif kind == 'lf':
            leaf_bits = int(tok[1])
        else:        
            # unknown value
            pass

        if not missing:
            # FIXME: pass leaf_bits
            builder.setdata(n_features, n_classes, leaf_bits)

    #print('load-model', leaves_found)
