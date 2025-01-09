
import os
import subprocess

import numpy
import pandas

import tensorflow as tf
from keras.datasets import mnist
import keras

# Make a simple CNN. Originally based on "mnist_arduino" TinyMaix example
def init_model(start=8, growth=1.5, classes=10):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Softmax, Activation, BatchNormalization, Dropout, DepthwiseConv2D
    from tensorflow.keras.layers import GlobalAveragePooling2D

    model = Sequential()
    g = growth

    model.add(Conv2D(start, (3,3), padding = 'valid',strides = (2, 2), input_shape = (28, 28, 1), name='ftr0'))
    model.add(BatchNormalization(name="bn0"))
    model.add(Activation('relu', name="relu0"))

    model.add(Conv2D(int(start*g), (3,3), padding = 'valid',strides = (2, 2), name='ftr1'))
    model.add(BatchNormalization(name="bn1"));
    model.add(Activation('relu',name="relu1"));

    model.add(Conv2D(int(start*g*g), (3,3), padding = 'valid',strides = (2, 2), name='ftr2'))
    model.add(BatchNormalization());
    model.add(Activation('relu')); 
    
    model.add(GlobalAveragePooling2D(name='GAP'))
    model.add(Dense(classes, name="fc1"))
    model.add(Activation('softmax', name="sm"))
    return model


def train_mnist(h5_file, epochs=10):
    (x_orig_train, y_orig_train), (x_orig_test, y_orig_test) = mnist.load_data() 
    num_classes = 10

    TEST_DATA_DIR = 'test_data'
    generate_test_files(TEST_DATA_DIR, x_orig_test, y_orig_test)
    print('Wrote test data to', TEST_DATA_DIR)

    x_train = x_orig_train
    x_test = x_orig_test
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255

    y_train = keras.utils.to_categorical(y_orig_train, num_classes)
    y_test = keras.utils.to_categorical(y_orig_test, num_classes)

    model = init_model(start=8)  
    model.summary()

    model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics = ["categorical_accuracy"]) 
    H = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data = (x_test, y_test), shuffle=True)

    model.save(h5_file)

def generate_test_files(out_dir, x, y, samples_per_class=5):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    expect_bytes = 28*28*1
    classes = numpy.unique(y)
    X_series = pandas.Series([s for s in x])
    Y_classes = y #numpy.argmax(y, axis=1)

    # select one per class
    for class_no in classes:
        matches = (Y_classes == class_no)
        x_matches = X_series[matches]

        selected = x_matches.sample(n=samples_per_class, random_state=1)
        for i, sample in enumerate(selected):
            out = os.path.join(out_dir, f'mnist_example_{i}_{class_no}.bin')
            data = sample.tobytes(order='C')

            assert len(data) == expect_bytes, (len(data), expect_bytes)
            with open(out, 'wb') as f:
                f.write(data)


def generate_tinymaix_model(h5_file,
        out_file : str,
        input_shape : tuple[int],
        output_shape : tuple[int],
        tools_dir,
        python_bin='python',
        precision='fp32',
        quantize_data=None,
        quantize_type='0to1',
    ):

    output_dequantize = quantize_data is not None

    # Convert .h5 to .tflite file
    assert h5_file.endswith('.h5'), 'Keras model HDF5 file must end with .h5'
    tflite_file = out_file + '.tflite'

    args = [
        python_bin,
        os.path.join(tools_dir, 'h5_to_tflite.py'),
        h5_file,
        tflite_file,
    ]
    if quantize_data is None:
        args += [ '0' ] 
    else:
        args += [
            '1',
            quantize_data,
            quantize_type,
        ]

    cmd = ' '.join(args)
    print('RUN', cmd)
    out = subprocess.check_output(args).decode('utf-8')

    # check that outputs have been created
    assert os.path.exists(tflite_file), tflite_file
    
    def format_shape(t : tuple[int]):
        return ','.join(str(i) for i in t)


    # Convert .tflite file to TinyMaix 
    tmld_file = tflite_file.replace('.tflite', '.tmdl')
    header_file = tmld_file.replace('.tmdl', '.h')
    args = [
        python_bin,
        os.path.join(tools_dir, 'tflite2tmdl.py'),
        tflite_file,
        tmld_file,
        precision,
        str(1 if output_dequantize else 0),
        format_shape(input_shape),
        format_shape(output_shape),
        #endian, #"<" or ">"
    ]
    cmd = ' '.join(args)
    print('RUN', cmd)
    subprocess.check_output(args).decode('utf-8')

    # check that outputs have been created
    assert os.path.exists(tmld_file), tmld_file
    assert os.path.exists(header_file), header_file

    return tmld_file

def main():

    h5_file = "mnist_cnn.h5"
    tinymaix_tools_dir = '../../dependencies/TinyMaix/tools'
    assert os.path.exists(tinymaix_tools_dir), tinymaix_tools_dir

    # Run training
    train_mnist(h5_file)

    # Export the model using TinyMaix
    # both with quantization and without
    for config in ('int8', 'fp32'):
        if config == 'int8':
            quantize_data = os.path.join(tinymaix_tools_dir, 'quant_img_mnist/')
        else:
            quantize_data = None # disables quantization

        out = generate_tinymaix_model(h5_file,
            out_file=h5_file.replace('.h5', '')+f'_{config}',
            input_shape=(28,28,1),
            output_shape=(1,),
            tools_dir=tinymaix_tools_dir,
            precision=config,
            quantize_data=quantize_data,
        )
        print('Wrote model to', out)

if __name__ == '__main__':
    main()
