
import array
import tinymaix_cnn

MNIST_MODEL = 'mnist_arduino_custom.tmdl'

def test_cnn_create():

    model = None
    with open(MNIST_MODEL, 'rb') as f:
        model_data = array.array('b', f.read())
        model = tinymaix_cnn.new(model_data)

        # TODO: enable these checks
        #wrong_type = array.array('f', [])
        #model.run(wrong_type)

        # TODO: enable these checks
        #wrong_size = array.array('B', [])
        #model.run(wrong_size)

    del model


def test_cnn_mnist():

    model = None
    with open(MNIST_MODEL, 'rb') as f:
        model_data = f.read()
        model = tinymaix_cnn.new(model_data)

    for class_no in range(0, 10):
        with open('mnist_example%d.bin'.format(class_no), 'rb') as f:
            img = f.read()
            out = model.run(img)
            # TODO replace with assert
            print(class_no, out, class_no == out)


test_cnn_create()
test_cnn_mnist()
