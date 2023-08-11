# device/micropython code

import emltrees
import array

model = emltrees.new(5, 30)

# Load a CSV file with the model
with open('xor_model.csv', 'r') as f:
    emltrees.load_model(model, f)

# run it
examples = [
    array.array('f', [0.0, 0.0]),
    array.array('f', [1.0, 1.0]),
    array.array('f', [0.0, 1.0]),
    array.array('f', [1.0, 0.0]),
]

for ex in examples: 
    result = model.predict(ex)
    print(ex, result)

