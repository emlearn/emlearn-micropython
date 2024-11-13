
# Human Activity Recognition with tree-based models

*Human Activity Recognition* (HAR) is the task of detecting what activities a human is performing based on motion.
This type of activity recognition is using an Inertial Measurement Unit (IMU) carried on the person.
The IMU consists has at least one accelerometer, gyro or magnetometer - or a combination of these.
It can be a smartphone, smartwatch, fitness tracker, or more specialized device.
The activities can for example be general daily activities, domestic activities, excercises, et.c.

The same kind of approach can also be applied to animals (*Animal Activity Recognition*),
which enables tracking of pets, lifestock and wild animals.
This has been used for many kinds of animals - such as cats, dogs, diary cattle.

The same approach can be used for simple gesture recognition.

## Status
Working. Tested running on ESP32 with MicroPython 1.24.

**NOTE:** This is primarily *example* code for a Human Activity Recognition,
not a generally-useful pretrained model.
The dataset used is rather simple, and may not reflect the data you get from your device
- which will lead to poor classifications.
For a real world usage you should probably replace the dataset with your own data, collected on your own device.

## Machine Learning pipeline

This example uses an approach based on the paper
[Are Microcontrollers Ready for Deep Learning-Based Human Activity Recognition?](https://www.mdpi.com/2079-9292/10/21/2640).
For each time-window, time-based statistical features are computed,
and then classified with a RandomForest model.

## Dataset
The example uses the [UCI-HAR dataset](https://www.archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions).
The classes are by default limited to the three static postures (standing, sitting, lying) plus three dynamic activities (walking, walking downstairs, walking upstairs).
The data is from a waist-mounted smartphone.
Samplerate is 50Hz.
By default only the accelerometer data is used (not the gyro).


## TODO

- Add an illustrative image
- Run the training + test/evaluation in CI
- Add demonstration on LilyGo T-Watch 2020 (BMA423 accelerometer)


## Running on host

To run the example on your PC using Unix port of MicroPython.

Make sure to have the Unix port of MicroPython setup.
On Windows you can use Windows Subsystem for Linux (WSL), or Docker.

Download the files in this example directory
```
git clone https://github.com/emlearn/emlearn-micropython.git
cd emlearn-micropython/examples/har_trees
```

Install the dependencies
```console
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_trees.mpy

micropython har_run.py
```

## Running on device (Viper IDE)

Make sure to have MicroPython installed on device.

The fastest and easiest to to install on your device is to use Viper IDE.
This will install the library and the example code:
[<img src="https://raw.githubusercontent.com/vshymanskyy/ViperIDE/refs/heads/main/assets/btn_run.png" alt="Run using ViperIDE" height="42"/>](https://viper-ide.org/?install=github:emlearn/emlearn-micropython/examples/har_trees)



## Run on device

Make sure to have MicroPython installed on device.

Install the dependencies
```console
mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/master/xtensawin_6.3/emlearn_trees.mpy
mpremote mip install github:jonnor/micropython-npyfile
mpremote mip install github:jonnor/micropython-zipfile
```

Copy example code
```
mpremote cp har_uci_trees.csv har_uci.testdata.npz timebased.py:
```

Run model evaluation on a test set
```
micropython har_run.py
```

## Run on device (with live accelerometer data)

FIXME: document

Requires a M5StickC PLUS 2.
Using a MPU6886 accelerometer connected via I2C.

`har_live.py`


## Run training

This will train a new model.
Uses CPython on the PC.

Install requirements
```
pip install -r requirements.txt
```

Download training data (4 GB space)
```
python -m leaf_clustering.data.har.uci
```

Run training process
```
python har_train.py
```

This will output a new model (named `_trees.csv`).
Can then be deployed to device following the steps above.


