
How ready is MicroPython for use in a "TinyML" setting,
building applications that utilize Machine Learning models on embedded/microcontroller device.
Since these are typically sensor-centric, data-intensive, connected applications -
the below exposition may also be useful in evaluating related cases.

NOTE: This is a working document intended to inform development.
Because of this the selection of is inherently somewhat pessimistic:
things that are lacking are more likely to appear rather than those that just-work.

Overall our belief is that MicroPython is the most promising
development platform for TinyML application development,
and that it has many of the key properties needed to be highly suitable for this purpose.

## Native modules

FIXED. Completely broken on at least one port, due to module functions being garbage collected.
https://github.com/micropython/micropython/issues/6769

Using floating point / math operations is difficult / very tedious.
Need to manually resolve all symbols.
https://github.com/micropython/micropython/issues/5629 

## Efficient data processing

#### Using specialized code emitters

Native machine code emitters with @micropython.native/viper.
Good! Integer only code can be made quite fast using this.

Limitation. Floating point code cannot be optimized with `@micropython.native`.

#### Using array.array

Support for fixed buffers, using `array.array`. OK!

!Very little documentation about using array.array.
What the benefits are, why to use it over lists etc, when it is critical.
With or without `@micropython.native`.

? Not easy to write allocation free code in Python for arrays.
Slicing notably makes a copy, whether it is for just reading or writing.
Missing an equivalent of a `memoryview` for arrays?

Inefficient conversion from bytes to `array.array`?
! Missing a "cast" type operation.
https://github.com/micropython/micropython/issues/4064
Typical usecase: When data comes an IMU/accelerometer over I2C/SPI it is stored as bytes.
However these actually represent integer values, typically int16,
stored in either little or big endian formats.
TODO: benchmark the various options here.
Plain buffer/array accesses (with native/viper) vs `struct.unpack` / `struct.unpack_into`.

!Inefficient creation of `array.array`.
Must use a generator-based constructor - the only type that is supported now.
This can take seconds for many items.
Would want to initialize empty / with unspecified values, or initialized filled with a particular value (like 0).

Inefficient copies of `array.array`.

## Data formats

#### Binary encoding into strings
Useful for serializing binary data for sending in textual protocols.
Be it USART, JSON based web API, etc.

base64 support is in core library. OK.
But encoding base64 cannot be done without allocations.
https://github.com/micropython/micropython/issues/15513

### Numpy .npy files

For storing multi-dimensional arrays. Basic compression support using ZIP.
Efficient implementation available in [micropython-npyfile](https://github.com/jonnor/micropython-npyfile), including compression with [micropython-zipfile](https://github.com/jonnor/micropython-zipfile). Good

### Audio files
!No mip-installable library for *.wav* files.
Just various example code lying around at various locations.
Note that CPython defines the API for a wavefile module.
Ideally would be compatible with that.

?No ready-to-install libraries for OPUS/MP3 encoding/decoding.

### Image files

!No mip-installable library for JPEG files.

!No mip installable library for PNG files.

OpenMV [omv.image](https://docs.openmv.io/library/omv.image.html) module supports loading/saving JPEG/PNG.

## Library interoperability

No interoperable datastructure for multi-dimensional arrays.
Many projects have invented their own, incompatible, representations.
Ref notes on [multi_dimensional_arrays](multi_dimensional_arrays.md).

## Drivers

#### IMU/Accelerometer/gyro
Almost no accelerometer/IMU drivers implement FIFO based-readout.
Causes poor sampling accuracy / high jitter / limited samplerate / high power consumption.
Can be changed by driver authors. [Discussion thread](https://github.com/orgs/micropython/discussions/15512).

#### Microphone

`I2S` is the only digital audio protocol with OK support.
However, I2S microphones is the more rare variety, PDM is more common.

!PDM microphones are not supported, on any port?
Bad, limits available ready-to-run hardware.
ESP32, RP2040, STM32, NRF52.

Microphone support also missing on `Unix` port.
Useful to support ready-to-run examples on host PC, testing during development.
And is relevant for deploying on Linux SBC such as Raspberry PI.
Could this be implemented as a .mpy native module?

#### Camera
No standard interface in upstream MicroPython.
Open issue: https://github.com/micropython/micropython/issues/15753
With proof-of-concept implementation for ESP32.

OpenMV has a good module, with drivers for dozens of popular cameras.
But is a custom distribution of MicroPython.

## Connectivity

#### HTTPS
Good support for WiFi based commmunication on ESP32.
Ethernet based support also seems workable on ESP32.

MQTT seems to be a bit chaotic.
Multiple competing libraries.
!?Not sure if any are really production grade.

#### Bluetooth Low Energy
Basic BLE connectivity also seems OK on ESP32.

Not good! *Low-power* BLE support.

ESP32 no support for sleep with Bluetooth.
https://github.com/micropython/micropython/pull/8318

NRF52 BLE support not so good? 
https://github.com/micropython/micropython/issues/9851
https://github.com/micropython/micropython/pull/8318

Missing! Support for Bluetooth Audio.
Useful for collecting raw data by sending to a phone or computer.

#### LoRa
Unknown/not investigated.

#### Zigbee
Unknown/not investigated.


## Feature Engineering / Preprocessing

For many Machine Learning use-cases.

### Filters

FIR filters.
`numpy.convolve` in [ulab](https://github.com/v923z/micropython-ulab) can probably be used?

IIR filters.
`scipy.signal.sosfilt` available in [ulab](https://github.com/v923z/micropython-ulab).
`emlearn_iir` available in [emlearn-micropython](https://github.com/emlearn/emlearn-micropython).

### Fast Fourier Transform (FFT)
Key part of computing frequency spectrum, or time-frequency representations (spectrogram).

FFT.
`numpy.fft.fft` available in [ulab](https://github.com/v923z/micropython-ulab).
`emlearn_fft` available in [emlearn-micropython](https://github.com/emlearn/emlearn-micropython).

DCT.
Not available?

## Machine Learning inference

### Linear models

!not available.
Logistic regression.
Support Vector Machine. Especially linear+binary.

### Tree-based ensembles (Random Forest / Decision Trees)

`emltrees` available in [emlearn-micropython](https://github.com/emlearn/emlearn-micropython).
Can alternatively generate Python code with everywhereml/m2cgen.
But over 10x slower than emlearn-micropython.

### K-nearest-neighbours

`emlearn_neighbors` available in [emlearn-micropython](https://github.com/emlearn/emlearn-micropython).

### Convolutional Neural Network

`tinymaix_cnn` available in [emlearn-micropython](https://github.com/emlearn/emlearn-micropython).

TensorFlow Lite support in [OpenMV](https://docs.openmv.io/library/omv.ml.html).
But as a custom MicroPython distribution.


