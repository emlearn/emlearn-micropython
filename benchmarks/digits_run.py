
import time

from digits_data import digits

from everywhere_digits import RandomForestClassifier
import m2c_digits
import emltrees


def load_model(builder, f):

    for line in f:
        line = line.rstrip('\r')
        line = line.rstrip('\n')
        tok = line.split(',')
        kind = tok[0]
        if kind == 'r':
            root = int(tok[1])
            emltrees.addroot(builder, root)
        elif kind == 'n':
            feature = int(tok[1])
            value = float(tok[2])
            left = int(tok[3])
            right = int(tok[4])
            emltrees.addnode(builder, left, right, feature, value)
        else:        
            # unknown value
            pass

def emlearn_create():
    model = emltrees.open(7, 150)

    # Load a CSV file with the model
    with open('eml_digits.csv', 'r') as f:
        load_model(model, f)
    return model

def everywhere_run(data):
    clf = RandomForestClassifier()
    for idx, x in enumerate(data):
        out = clf.predict(x)
        print('exptected %d, got %d' % (idx, out))

def emlearn_run(data):

    model = emlearn_create()
    for idx, x in enumerate(data):
        f = array.array('f', x)
        out = emltrees.predict(model, f)
        print('exptected %d, got %d' % (idx, out))    

def m2c_run(data):

    for idx, x in enumerate(data):
        out = m2c_digits.score(x)
        print('exptected %d, got %d' % (idx, out))

data = digits
emlearn_run(data)
everywhere_run(data)
m2c_run(data)
