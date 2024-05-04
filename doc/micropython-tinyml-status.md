
How ready is MicroPython for use in a TinyML setting.

## Efficient data processing

Good! Integer only code can be made quite fast with @micropython.native/viper

Missing documentation about using array.array
With or without @micropython.native

Floating point code cannot be optimized with @micropython.native

## Interoperability

No interoperable datastructure for multi-dimensional arrays

## Drivers

Almost no accelerometer/IMU drivers implement FIFO based-readout.
Causes poor sampling accuracy / high jitter.

PDM microphones are not supported.
ESP32
RP2040

## Preprocessing

FIR filters.

IIR filters.

FFT.

## Machine Learning inference

Logistic regression

Support Vector Machine. Especially linear+binary

Tree-based ensembles (Random Forest / Decision Trees)

K-nearest-neighbours

Convolutional Neural Network


