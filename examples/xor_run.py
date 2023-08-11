# device/micropython code

import emltrees
import array

# create
builder = emltrees.open(5, 30)

# Load a CSV file with the model
with open('xor_model.csv', 'r') as f:
    emltrees.load_model(builder, f)

# run it
examples = [
    array.array('f', [0.0, 0.0]),
    array.array('f', [1.0, 1.0]),
    array.array('f', [0.0, 1.0]),
    array.array('f', [1.0, 0.0]),
]

for ex in examples: 
    result = emltrees.predict(builder, ex)
    print(ex, result)

