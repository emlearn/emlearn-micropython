
import emlearn_fft

import array
import gc

def test_fft_del():
    """
    Deleting should free all the memory
    """
    gc.enable()
    gc.collect()
    before_new = gc.mem_alloc()

    fft_length = 256

    model = emlearn_fft.FFT(fft_length)
    after_new = gc.mem_alloc()
    added = after_new - before_new

    assert added > 1000, added
    assert added < 2000, added
    del model
    gc.collect()
    after_del = gc.mem_alloc()
    diff = after_del - before_new
    print(before_new, after_new, after_del)
    assert diff == 0, diff


def test_fft_run():
    """
    Running should give output
    """

    fft_length = 128
    in_real = array.array('f', (0.0 for _ in range(fft_length)))
    in_imag = array.array('f', (0.0 for _ in range(fft_length)))
    model = emlearn_fft.FFT(fft_length)

    emlearn_fft.fill(model, fft_length)
    model.run(in_real, in_imag)

    # FIXME: use some reasonable input data and assert the output data

test_fft_del()
test_fft_run()
