
# Developing emlearn-micropython

For those that wish to hack on emlearn-micropython itself.

Contribution guidelines can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

If you just wish to use it as a library, see instead the [usage guide](https://emlearn-micropython.readthedocs.io/en/latest/user_guide.html).

#### Prerequisites

These instructions have been tested on Linux.
They might also work on Mac OS.
For Windows, I recommend using Windows Subsystem for Linux (WSL2).

You will need to have **Python 3.10+ or later** already installed.

We assume that **micropython** git repository available.
It is assumed to be at the same level as this repository in the file system.
If using another location, adjust `MPY_DIR` accordingly.
You should be using MicroPython 1.25 (or newer).

You should build and install the [MicroPython Unix port](https://github.com/micropython/micropython/blob/master/ports/unix/README.md) to run/test on PC (`micropython` executable).

To build and test on device, make sure you have the **build toolchain** needed for your hardware platform.
See [MicroPython: Building native modules](https://docs.micropython.org/en/latest/develop/natmod.html),
and the documentation for the MicroPython port/architecture of interest.
For example `esp32`, `stm32` or `rp2`.


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

NOTE: Requires `micropython` program to installed (MicroPython Unix port).

To build and run tests on host
```
make check
```


#### Build for device

NOTE: Requires the toolchain for the particular device to be installed.
See MicroPython documentation for the port in-question.

Build the .mpy native module
```
make dist ARCH=armv6m MPY_DIR=../micropython
```

Install it on device
```
mpremote cp dist/armv6m*/emlearn_trees.mpy :emlearn_trees.mpy
```


## Building documentation

Make sure to have dev dependencies

```
pip install -r requirements.dev.txt
```

Run the Sphinx build

```
make -C docs/ html
```

Open the frontpage in browser `docs/_build/html/index.html`

