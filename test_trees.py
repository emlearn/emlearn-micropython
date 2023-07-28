
import emltrees
import array

"""
Code to export emlearn model as CSV forma

def save_csv(trees):
    nodes, roots = trees.forest_
    nodes = nodes.copy()
    lines = []
    for r in roots:
        lines.append(f'r,{r}')
    for n in nodes:
        lines.append(f'n,{n[0]},{n[1]},{n[2]},{n[3]}')
    code = '\r\n'.join(lines) 
    return code
"""

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

# create
builder = emltrees.open(5, 30)

# TODO: read a CSV file with the model
with open('xor_model.csv', 'r') as f:
    load_model(builder, f)

# run it
ex1 = array.array('f', [0.0, 1.0])
ex2 = array.array('f', [0.0, 0.0])
ex3 = array.array('f', [1.0, 0.0])

print(emltrees.predict(builder, ex1))
print(emltrees.predict(builder, ex2))
print(emltrees.predict(builder, ex3))

