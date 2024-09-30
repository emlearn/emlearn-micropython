
import array

# FIXME: scale should be integer that matches with running/instantiating IIR
def convert_coefficients(c, scale=(2**13)-1):
    out = array.array('h', (0 for _ in range(len(c))))
    assert (len(c) % 6 == 0), len(c)
    n_stages = len(c) // 6
    
    # Transposed Direct Form II / sosfilt / emliir
    # [ b0, b1, b2, 1.0, -a1, -a2 ]
    # Direct Form I / eml_iir_q15 
    # [ b0, 0, b1, b2, a1, a2 ]

    # NOTE: CMSIS-DSP “a” coefficients are negative compared to SciPy conventions
    for stage in range(n_stages):
        offset = stage*6
        b0, b1, b2, a0, a1, a2 = c[offset:offset+6]
        assert a0 == 1.0
        out[offset+0] = int(b0 * scale)
        out[offset+1] = 0
        out[offset+2] = int(b1 * scale)
        out[offset+3] = int(b2 * scale)
        out[offset+4] = int(-a1 * scale)
        out[offset+5] = int(-a2 * scale)

    return out
