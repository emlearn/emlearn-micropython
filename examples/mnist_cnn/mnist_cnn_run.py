
import os
import array
import time
import gc

import emlearn_cnn_int8

MODEL = 'mnist_cnn_int8.tmdl'
TEST_DATA_DIR = 'test_data'

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

def load_images_from_directory(path):
    sep = '/'

    for filename in os.listdir(path):
        # TODO: support standard image formats, like .bmp/.png/.jpeg
        if not filename.endswith('.bin'):
            continue

        # Find the label (if any). The last part, X_label.format
        label = None
        basename = filename.split('.')[0]
        tok = basename.split('_')
        if len(tok) > 2:
            label = tok[-1]

        data_path = path + sep + filename
        with open(data_path, 'rb') as f:
            img = array.array('B', f.read())

            yield img, label

def test_cnn_mnist():

    # load model
    model = None
    with open(MODEL, 'rb') as f:
        model_data = array.array('B', f.read())
        model = emlearn_cnn.new(model_data)

    out_length = model.output_dimensions()[0]
    probabilities = array.array('f', (-1 for _ in range(out_length)))

    # run on some test data
    n_correct = 0
    n_total = 0
    for img, label in load_images_from_directory(TEST_DATA_DIR):
        class_no = int(label) # mnist class labels are digits

        #print_2d_buffer(img, 28)

        run_start = time.ticks_us()
        model.run(img, probabilities)
        out = argmax(probabilities)
        run_duration = time.ticks_diff(time.ticks_us(), run_start) / 1000.0 # ms
        correct = class_no == out
        n_total += 1
        if correct:
            n_correct += 1

        print('mnist-example-check', class_no, '=', out, correct, round(run_duration, 3))
        
        gc.collect()

    accuracy = n_correct / n_total
    print('mnist-example-done', n_correct, '/', n_total, round(accuracy*100, ), '%')

if __name__ == '__main__':
    test_cnn_mnist()
