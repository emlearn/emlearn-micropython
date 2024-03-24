

def load_model(builder, f):

    leaves_found = 0

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
            value = float(tok[2])
            left = int(tok[3])
            right = int(tok[4])
            builder.addnode(left, right, feature, value)
        elif kind == 'l':
            leaf = int(tok[1])
            builder.addleaf(leaf)
            leaves_found += 1
        else:        
            # unknown value
            pass

    #print('load-model', leaves_found)
