
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

    for line in f:
        line = line.rstrip('\r')
        line = line.rstrip('\n')
        tok = line.split(',')
        kind = tok[0]
        if kind == 'r':
            root = int(tok[1])
            builder.addroot(root)
        elif kind == 'n':
            feature = int(tok[1])
            value = int(float(tok[2]))
            left = int(tok[3])
            right = int(tok[4])
            builder.addnode(left, right, feature, value)
        elif kind == 'l':
            leaf = int(tok[1])
            builder.addleaf(leaf)
            leaves_found += 1
        elif kind == 'f':
            n_features = int(tok[1])
        elif kind == 'c':
            n_classes = int(tok[1])
        else:        
            # unknown value
            pass

    builder.setdata(n_features, n_classes)

    #print('load-model', leaves_found)
