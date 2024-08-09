
How ready is MicroPython for use in a TinyML setting.

## Native modules

Completely broken on at least one port, due to being garbage collected.

Using floating point / math operations is difficult / very tedious.
Need to manually resolve all symbols.
https://github.com/micropython/micropython/issues/5629 

## Efficient data processing

Good! Support for fixed buffers, using `array.array`.

!Missing documentation about using array.array.
With or without @micropython.native

? Not easy to write allocation free code in Python for arrays.
Slicing notably makes a copy, whether it is for just reading or writing.
Missing an equivalent of a memoryview for arrays?

Native machine code emitters with @micropython.native/viper
Good! Integer only code can be made quite fast with 
Limitation. Floating point code cannot be optimized with @micropython.native

Inefficient conversion from bytes to array.array?
! Missing a "cast" type operation.
When data comes an IMU/accelerometer over I2C/SPI it is just bytes.
However these actually represent integer values, typically int16,
stored in either little or big endian formats.
TODO: benchmark the various options here.
Plain buffer/array accesses vs struct.unpack / unpack_into

!Inefficient creation of array.array
Must use a generator-based constructor - the only type that is standardized.
This can take seconds for many items.
Would want to initialize , or initialized filled with a particular value.

Semi-related. Encoding base64 cannot be done without allocations. LINK issue
Useful for serializing binary data for sending in textual protocols.
Be it USART, JSON based web API, etc.

## Interoperability

No interoperable datastructure for multi-dimensional arrays.
Ref notes on [multi_dimensional_arrays](multi_dimensional_arrays.md).

## Drivers

Almost no accelerometer/IMU drivers implement FIFO based-readout.
Causes poor sampling accuracy / high jitter / limited samplerate / high power consumption.
Can be changed by driver authors. [Discussion thread](https://github.com/orgs/micropython/discussions/15512).

I2S is the only digital audio protocol with OK support.
However, I2S microphones is the more rare variety, PDM is more common.
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

Missing! Support for Bluetooth Audio.
Useful for collecting raw data by sending to a phone or computer.

## Preprocessing

FIR filters.
numpy.convolve in ulab can probably be used?

IIR filters.
sosfilt in ulab.

FFT.
Available in ulab.

DCT.
Not available?

## Machine Learning inference

Logistic regression

Support Vector Machine. Especially linear+binary

Tree-based ensembles (Random Forest / Decision Trees)

K-nearest-neighbours

Convolutional Neural Network


