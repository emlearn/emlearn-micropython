[![DOI](https://zenodo.org/badge/670384512.svg)](https://zenodo.org/badge/latestdoi/670384512)

# emlearn-micropython

Machine Learning and Digital Signal Processing for [MicroPython](https://micropython.org).
Provides convenient and efficient MicroPython modules, and enables MicroPython application developers
to run efficient Machine Learning models on microcontroller, without having to touch any C code.

This is a [TinyML](https://www.tinyml.org/) library,
particularly well suited for low-compexity and low-power classification tasks.
It can be combined with feature preprocessing, including neural networks to address more complex tasks.

Builds on [emlearn](https://emlearn.org), a C99 library for Machine Learning on microcontrollers and embedded system.

> scikit-learn for Microcontrollers

## Status
**Minimally useful**

- Initial set of Machine Learning and Digital Signal Processing modules available, including example projects.
- Supports most MicroPython ports using runtime installable native modules
- Primarily tested on `x64` (Unix port) and `xtensawin` (ESP32/ESP32-S3/etc).
- Some devices only supported using external, notably armv6m/Cortex-M0/RP2040
- Not yet supported: [RISC-V/ESP32-C3/ESP32-C6](https://github.com/emlearn/emlearn-micropython/issues/35)

## Features

- Classification with [RandomForest](https://en.wikipedia.org/wiki/Random_forest)/DecisionTree models
- Classification and on-device learning with [K-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- Classification with Convolutional Neural Network (CNN), using [TinyMaix](https://github.com/sipeed/TinyMaix/) library.
- [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform) for feature preprocessing, or general DSP
- Infinite Impulse Response (IIR) filters for feature preprocessing, or general DSP
- Clustering using K-means
- Scaling and data type transformations for `array`, using `emlearn_arrayutils`.
- Load/save Numpy .npy files using [micropython-npyfile](https://github.com/jonnor/micropython-npyfile/)
- Installable as a MicroPython native module. No rebuild/flashing needed
- Operates on standard `array.array` data structures
- Models can be loaded at runtime from a file in disk/flash
- Highly efficient. Inference times down to 100 microseconds, RAM usage <2 kB, FLASH usage <2 kB
- Pre-built native modules available for most common architectures `xtensawin`.

## Examples

- [xor_trees](./examples/xor_trees/). A "Hello World", using RandomForest.
- [mnist_cnn](./examples/mnist_cnn/). Basic image classification, using Convolutional Neural Network.
- [har_trees](./examples/har_trees/). Accelerometer-based Human Activity Recognition, using Random Forest
- [soundlevel_iir](./examples/soundlevel_iir/). Sound Level Meter, using Infinite Impulse Response (IIR) filters.

## Documentation

Complete usage [documentation on ReadTheDocs](https://emlearn-micropython.readthedocs.io/en/latest/user_guide.html).


## Citations

If you use `emlearn-micropython` in an academic work, please reference it using:

```tex
@misc{emlearn_micropython,
  author       = {Jon Nordby},
  title        = {{emlearn-micropython: Efficient Machine Learning engine for MicroPython}},
  month        = aug,
  year         = 2023,
  doi          = {10.5281/zenodo.8212731},
  url          = {https://doi.org/10.5281/zenodo.8212731}
}
```



## Developing

For those that wish to hack on emlearn-micropython itself.

#### Download the code

Clone the repository using git
```
git clone https://github.com/emlearn/emlearn-micropython
```

#### Prerequisites
These come in addition to the prequisites described above.

Make sure you have the dependencies needed to build for your platform.
See [MicroPython: Building native modules](https://docs.micropython.org/en/latest/develop/natmod.html).

We assume that micropython is installed in the same place as this repository.
If using another location, adjust `MPY_DIR` accordingly.

You should be using MicroPython 1.25 (or newer).

#### Build

Build the .mpy native module
```
make dist ARCH=xtensawin MPY_DIR=../micropython
```

Install it on device
```
mpremote cp dist/armv6m*/emlearn_trees.mpy :emlearn_trees.mpy
```

#### Run tests

To build and run tests on host
```
make check
```


