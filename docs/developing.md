
# Developing emlearn-micropython

For those that wish to hack on emlearn-micropython itself.

If you just wish to use it as a library, see instead the [usage guide](https://emlearn-micropython.readthedocs.io/en/latest/user_guide.html).

#### Prerequisites

You will need to have **Python 3.10+ or later** already installed.

We assume that **micropython** git repository available.
It is assumed to be at the same level as this repository in the file system.
If using another location, adjust `MPY_DIR` accordingly.
You should be using MicroPython 1.25 (or newer).

Make sure you have the **build toolchain** needed for your platform.
See [MicroPython: Building native modules](https://docs.micropython.org/en/latest/develop/natmod.html),
and the documentation for the MicroPython port/architecture of interest.

#### Download the code

Clone the repository using git
```
git clone https://github.com/emlearn/emlearn-micropython
```

#### Download dependencies

Fetch git submodules

```
git submodule update --init
```

NOTE: Recommend using a Python virtual environment (using `venv`, `uv`, etc.)

Install Python packages
```
pip install -r requirements.txt
```


#### Run tests on PC

To build and run tests on host
```
make check
```


#### Build for device

Build the .mpy native module
```
make dist ARCH=armv6m MPY_DIR=../micropython
```

Install it on device
```
mpremote cp dist/armv6m*/emlearn_trees.mpy :emlearn_trees.mpy
```


