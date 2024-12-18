
import array
import emlearn_cnn
import time
import gc

MODEL = 'mnist_cnn.tmdl'
TEST_DATA_DIR = 'data/'

def argmax(arr):
    idx_max = 0
    value_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > value_max:
            value_max = arr[i]
            idx_max = i

    return idx_max

def print_2d_buffer(arr, rowstride):

    rows = len(arr) // rowstride
    columns = rowstride

    for r in range(rows):
        for c in range(columns):
            v = arr[(r*rowstride)+c]
            print('{v:03d}'.format(v=v), end='')

        gc.collect()
        print('\n')

def test_cnn_mnist():

    # load model
    model = None
    with open(MODEL, 'rb') as f:
        model_data = array.array('B', f.read())
        model = emlearn_cnn.new(model_data)

    out_length = model.output_dimensions()[0]
    probabilities = array.array('f', (-1 for _ in range(out_length)))

    # run on some test data
    for class_no in range(0, 10):
        data_path = TEST_DATA_DIR + 'mnist_example_{0:d}.bin'.format(class_no)
        #print('open', data_path)
        with open(data_path, 'rb') as f:
            img = array.array('B', f.read())

            print_2d_buffer(img, 28)

            run_start = time.ticks_us()
            model.run(img, probabilities)
            out = argmax(probabilities)
            run_duration = time.ticks_diff(time.ticks_us(), run_start) / 1000.0 # ms

            print('mnist-example-check', class_no, out, class_no == out, run_duration)
        
        gc.collect()

if __name__ == '__main__':
    test_cnn_mnist()
