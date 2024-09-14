# device/micropython code

import emltrees
import array

model = emltrees.new(5, 30, 2)

# Load a CSV file with the model
with open('xor_model.csv', 'r') as f:
    emltrees.load_model(model, f)

# run it
max_val = (2**15-1) # 1.0 as int16
examples = [
    array.array('h', [0, 0]),
    array.array('h', [max_val, max_val]),
    array.array('h', [0, max_val]),
    array.array('h', [max_val, 0]),
]

for ex in examples: 
    result = model.predict(ex)
    print(ex, result)

