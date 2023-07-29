
# emlearn-micropython

[Micropython](https://micropython.org) integration for the [emlearn](https://emlearn.org) Machine Learning library for microcontrollers.

The goal is to enable applications to run ML inference on the microcontroller,
without having to touch any C code.

## Status
**Proof of Concept**

- Can run RandomForest/DecisionTree
- Module must be built manually (no pre-built releases)
- Has been tested with the `unix` MicroPython port

Wishing the support was more production level?
Vote in the [issue tracker](http://github.com/emlearn/emlearn-micropython/issues/1).
Or contribute yourself!

## Prerequisites

Minimally you will need

- Python 3.10+ on host
- MicroPython running onto your device

If there is no ready-made build for your device/architecture,
then you will need to build the .mpy module yourself.
For that you need to install more dependencies,
see [MicroPython: Building native modules](https://docs.micropython.org/en/latest/develop/natmod.html).

## Installing from a release


```
mpi install FIXME-make-release-on-github ESP32/RP2040
```


## Building and installing locally


Build the .mpy native module
```
make -C eml_trees/ ARCH=x64 MPY_DIR=../../micropython
```

Install it on device
```
FIXME: document
```


### Usage

Train a model with scikit-learn
```
pip install emlearn scikit-learn
python examples/xor_train.py
```

Copy model file to device

```
```

Copy the main script to device

```
examples/xor_device.py
```


