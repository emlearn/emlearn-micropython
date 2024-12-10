
import skimage.io
import skimage.color
import skimage.transform
import numpy

p = 'data/rps-cv-images/rock/dnss2tOuxRmL0ZjZ.png'
a = skimage.io.imread(p)
print(a.shape, a.dtype)

o = skimage.color.rgb2gray(a)
o = skimage.transform.resize(o, (96, 96))
o = (o * 255).astype(numpy.uint8)

print(o.shape, o.dtype)

numpy.save('inp.npy', o, allow_pickle=False)
