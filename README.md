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
- Supports most MicroPython ports using runtime installable native modules (.mpy files)
- Can also be included into a custom MicroPython build using external C modules
- Primarily tested on `x64` (Unix port) and `xtensawin` (ESP32/ESP32-S3/etc).


## Features

- Classification with [RandomForest](https://en.wikipedia.org/wiki/Random_forest)/DecisionTree models
- Classification and on-device learning with [K-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- Classification with Convolutional Neural Network (CNN), using [TinyMaix](https://github.com/sipeed/TinyMaix/) library.
- Regression and on-device learning with Linear Regression
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
Good starting points:

- [Getting started on PC (Linux/MacOS/Windows)](https://emlearn-micropython.readthedocs.io/en/latest/getting_started_host.html)
- [Getting started on device (ESP32/RP2/STM32/etc)](https://emlearn-micropython.readthedocs.io/en/latest/getting_started_device.html)


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

For how to hack on emlearn-micropython itself, see [docs/developing.md](docs/developing.md),
and for how to contribute see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).


