
import tinymaix_cnn

def test_cnn_create():

    model = None
    with open('mnist.tmd', 'r') as f:
        model_data = f.read()
        model = tinymaix_cnn.new(model_data)

    del model


def test_cnn_mnist():

    model = None
    with open('mnist.tmd', 'r') as f:
        model_data = f.read()
        model = tinymaix_cnn.new(model_data)

    for class_no in range(0, 10):
        with open('mnist_example%d'.format(class_no), 'r') as f:
            img = f.read()
            out = model.run(img)
            # TODO replace with assert
            print(class_no, out, class_no == out)


test_cnn_create()
