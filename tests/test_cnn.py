
import array
import emlearn_cnn_int8
import emlearn_cnn_fp32

MNIST_MODEL_INT8 = 'examples/mnist_cnn/mnist_cnn_int8.tmdl'
MNIST_MODEL_FP32 = 'examples/mnist_cnn/mnist_cnn_fp32.tmdl'
MNIST_DATA_DIR = 'examples/mnist_cnn/data/'

def test_cnn_create():

    model = None
    with open(MNIST_MODEL_FP32, 'rb') as f:
        model_data = array.array('B', f.read())
        model = emlearn_cnn_fp32.new(model_data)

        out_shape = model.output_dimensions()
        assert out_shape == (10,), (out_shape)

        # TODO: enable these checks
        #wrong_type = array.array('f', [])
        #model.run(wrong_type)

        # TODO: enable these checks
        #wrong_size = array.array('B', [])
        #model.run(wrong_size)

    del model


def print_2d_buffer(arr, rowstride):

    rows = len(arr) // rowstride
    columns = rowstride

    for r in range(rows):
        for c in range(columns):
            v = arr[(r*rowstride)+c]
            print('{v:03d}'.format(v=v), end='')

        print('\n')

def argmax(arr):
    idx_max = 0
    value_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > value_max:
            value_max = arr[i]
            idx_max = i

    return idx_max

def check_cnn_mnist(cnn_module, model_path):

    model = None
    with open(model_path, 'rb') as f:
        model_data = array.array('B', f.read())
        model = cnn_module.new(model_data)

    probabilities = array.array('f', (-1 for _ in range(10)))

    correct = 0
    for class_no in range(0, 10):
        data_path = MNIST_DATA_DIR + 'mnist_example_{0:d}.bin'.format(class_no)
        #print('open', data_path)
        with open(data_path, 'rb') as f:
            img = array.array('B', f.read())

            #print_2d_buffer(img, 28)

            model.run(img, probabilities)
            out = argmax(probabilities)
            # TODO replace with assert
            print('mnist-example-check', class_no, out, class_no == out)
            if out == class_no:
                correct += 1            

    assert correct >= 9, correct

def test_cnn_mnist_int8():
    check_cnn_mnist(emlearn_cnn_int8, MNIST_MODEL_INT8)


def test_cnn_mnist_fp32():
    check_cnn_mnist(emlearn_cnn_fp32, MNIST_MODEL_FP32)
    

test_cnn_create()
test_cnn_mnist_int8()
test_cnn_mnist_fp32()
