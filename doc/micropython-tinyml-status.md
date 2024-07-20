
How ready is MicroPython for use in a TinyML setting.

##


## Efficient data processing

Good! Support for fixed buffers, using `array.array`.

!Missing documentation about using array.array.
With or without @micropython.native

? Not easy to write allocation free code in Python for buffers/arrays.
Slicing is

Native machine code emitters with @micropython.native/viper
Good! Integer only code can be made quite fast with 
Limitation. Floating point code cannot be optimized with @micropython.native


## Interoperability

No interoperable datastructure for multi-dimensional arrays.
Ref notes on [multi_dimensional_arrays](multi_dimensional_arrays.md).

## Drivers

Almost no accelerometer/IMU drivers implement FIFO based-readout.
Causes poor sampling accuracy / high jitter / limited samplerate / high power consumption.
Details in notes.

PDM microphones are not supported, on any port?
ESP32, RP2040, STM32, NRF52.

## Connectivity

Good support for WiFi based commmunication on ESP32.

Not good! Low-power BLE support.

ESP32 no support for sleep with Bluetooth.
https://github.com/micropython/micropython/pull/8318

NRF52 BLE support not so good? 
https://github.com/micropython/micropython/issues/9851
https://github.com/micropython/micropython/pull/8318

LoRa?
Zigbee?

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


