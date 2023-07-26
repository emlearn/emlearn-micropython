

import emltrees
import array

res = emltrees.factorial(3)
print(res)

builder = emltrees.open(3, 10)
print(builder)

# TODO: read a CSV file with the model
emltrees.addnode(builder, 0, 1, 2, 3.14, 1)


ex1 = array.array('f', [200.0, 150.0, 300.0])
ex2 = array.array('f', [100.0, 450.0, 100.0])

print(emltrees.predict(builder, ex1))

print(emltrees.predict(builder, ex2))

