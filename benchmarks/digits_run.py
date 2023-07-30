
import time
import array

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

def argmax(l):
    max_value = l[0]
    max_idx = 0
    for i, v in enumerate(l):
        if v > max_value:
            max_value = v
            max_idx = i
    return max_idx


clf = RandomForestClassifier()
def everywhere_run(data):
    errors = 0
    for idx, x in enumerate(data):
        #x = array.array('f', x)
        out = clf.predict(x)
        if (idx != out):
            errors += 1
    return errors

model = emlearn_create()
def emlearn_run(data):
    errors = 0
    for idx, x in enumerate(data):
        f = array.array('f', x) # NOTE: this takes as long as predict
        out = emltrees.predict(model, f)
        if (idx != out):
            errors += 1
    return errors

def m2c_run(data):
    errors = 0
    for idx, x in enumerate(data):
        #x = array.array('f', x)
        scores = m2c_digits.score(x)
        out = argmax(scores)
        if (idx != out):
            errors += 1
    return errors

def none_run(data):
    errors = 0
    for idx, x in enumerate(data):
        tmp = len(x)
        out = idx
        if (idx != out):
            errors += 1
    return errors

def benchmark():

    data = digits

    print('model,errors,time_us')

    before = time.ticks_us()
    none_errors = none_run(data)
    after = time.ticks_us()
    none_duration = time.ticks_diff(after, before)
    print('none,{},{}'.format(none_errors, none_duration))

    before = time.ticks_us()
    eml_errors = emlearn_run(data)
    after = time.ticks_us()
    eml_duration = time.ticks_diff(after, before)
    print('emlearn,{},{}'.format(eml_errors, eml_duration))

    before = time.ticks_us()
    everywhere_errors = everywhere_run(data)
    after = time.ticks_us()
    everywhere_duration = time.ticks_diff(after, before)
    print('everywhere,{},{}'.format(everywhere_errors, everywhere_duration))

    before = time.ticks_us()
    m2c_errors = m2c_run(data)
    after = time.ticks_us()
    m2c_duration = time.ticks_diff(after, before)
    print('m2cgen,{},{}'.format(m2c_errors, m2c_duration))


