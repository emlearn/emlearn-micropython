# device/micropython code

import emlearn
import array

def load_model(builder, f):

    for line in f:
        line = line.rstrip('\r')
        line = line.rstrip('\n')
        tok = line.split(',')
        kind = tok[0]
        if kind == 'r':
            root = int(tok[1])
            emlearn.addroot(builder, root)
        elif kind == 'n':
            feature = int(tok[1])
            value = float(tok[2])
            left = int(tok[3])
            right = int(tok[4])
            emlearn.addnode(builder, left, right, feature, value)
        else:        
            # unknown value
            pass

# create
builder = emlearn.open(5, 30)

# Load a CSV file with the model
with open('xor_model.csv', 'r') as f:
    load_model(builder, f)

# run it
examples = [
    array.array('f', [0.0, 0.0]),
    array.array('f', [1.0, 1.0]),
    array.array('f', [0.0, 1.0]),
    array.array('f', [1.0, 0.0]),
]

for ex in examples: 
    result = emlearn.predict(builder, ex)
    print(ex, result)

