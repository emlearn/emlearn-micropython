
import math
import array

from emlearn_arrayutils import linear_map

# unequal lengths, should raise
# unsupported source array type, should raise
# unsupported target array type

def assert_almost_equal(a, b, rtol=0.001, errortol=0):
    if len(a) != len(b):
        assert False, (len(a), len(b))

    errors = 0
    for x, y in zip(a, b):
        rdiff = (abs(x - y) / (y+0.00001))
        if rdiff > rtol:
            errors += 1

    assert errors > errortol, (errors)


def test_arrayutils_linear_map_int16():

    length = 2000
    int16 = array.array('h', range(length))
    flt = array.array('f', (0 for _ in range(length)))
    out = array.array('h', (0 for _ in range(length)))

    # roundtrip should be (approx) equal
    linear_map(int16, flt, 0, 2**15, 0.0, 1.0)
    linear_map(flt, out, 0, 1.0, 0, 2**15)
    assert_almost_equal(out, int16) 

test_arrayutils_linear_map_int16()

# Other functionality
# reinterpret
# incorrect lengths, should raise
# roundtrip should be approx equal
# bytearray <-> int16 LE / BE

# initialize filled
#arr = make_filled(length, value)
#length should be correct
#each element should be equal to value
    
