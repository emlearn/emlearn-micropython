
# Image classification using Convolutional Neural Network


## Install requirements

Make sure to have Python **3.10** installed.
At the time TinyMaix, used for CNN, only supports Tensorflow <2.14,
which is not supported on Python 3.12 or later.

Make sure to have the Unix port of MicroPython 1.23 setup.
On Windows you can use Windows Subsystem for Linux (WSL), or Docker.

Install the dependencies of this example:
```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run training

This will train a Convolutional Neural Network using Keras.
The process will output `.tmdl` that is our built model.

```console
python mnist_train.py
```

## Running on host

```console
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_cnn.mpy

micropython mnist_cnn_run.py
```

## Running on device (ViperIDE)

The fastest and easiest to to install on your device is to use Viper IDE.
This will install the library and the example code:
[<img src="https://raw.githubusercontent.com/vshymanskyy/ViperIDE/refs/heads/main/assets/btn_run.png" alt="Run using ViperIDE" height="42"/>](https://viper-ide.org/?install=github:emlearn/emlearn-micropython/examples/mnist_cnn)


## Running on device (manually)

!Make sure you have it running successfully on host first.

Flash your device with a standard MicroPython firmware,
from the MicroPython.org downloads page.

```console
mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/master/xtensawin_6.3/emlearn_cnn.mpy
```

```console
mpremote cp mnist_cnn.tmdl :
mpremote cp -r data/ :
mpremote run mnist_cnn_run.py
```

## Running with live camera input

This example requires hardware with an ESP32 chip, and an OV2640 camera module.
It has been tested on Lilygo T-Camera Mic v1.6. 

You need to build a custom ESP32 firmware,
and add this C module for camera access:
https://github.com/cnadler86/micropython-camera-API

```
mpremote run mnist_cnn_camera.py
```

