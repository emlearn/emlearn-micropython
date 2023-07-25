
import emltrees
import array


f = array.array('f', [200.0, 150.0, 300.0])
r = emltrees.predict(f)
print(r)
