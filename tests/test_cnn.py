
import array
import tinymaix_cnn

MNIST_MODEL = 'mnist_arduino_custom.tmdl'

def test_cnn_create():

    model = None
    with open(MNIST_MODEL, 'rb') as f:
        model_data = array.array('B', f.read())
        model = tinymaix_cnn.new(model_data)

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

def test_cnn_mnist():

    model = None
    with open(MNIST_MODEL, 'rb') as f:
        model_data = array.array('B', f.read())
        model = tinymaix_cnn.new(model_data)

    for class_no in range(0, 10):
        data_path = 'test_data/mnist_example_{0:d}.bin'.format(class_no)
        print('open', data_path)
        with open(data_path, 'rb') as f:
            img = array.array('B', f.read())

            print_2d_buffer(img, 28)

            out = model.run(img)
            # TODO replace with assert
            print(class_no, out, class_no == out)


test_cnn_create()
test_cnn_mnist()
