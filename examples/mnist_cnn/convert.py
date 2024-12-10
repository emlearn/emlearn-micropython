
import sys
import numpy
import skimage.io

p = sys.argv[1]
o = sys.argv[2]
a = numpy.load(p)
print(a.shape)

skimage.io.imsave(o, a)
