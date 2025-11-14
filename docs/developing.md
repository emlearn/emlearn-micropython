
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
You should be using MicroPython 1.26 (or newer).

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

#### Run tests on PC using external modules build

This runs tests by building the modules as external C modules,
which bakes them into the firmware image/executable.

NOTE: Tested on Linux and Mac OS. Not tested on Windows Subsystem for Linux (WSL).

```
make check_unix
```

You should see each of the test functions in tests/ being ran,
and then a summary at the end with something like:

```
...
Passed: 17
Failed: 0
```

#### Run tests on PC using dynamic native modules

This runs tests by building as dynamic native modules (.mpy files),
and the .mpy files are then loaded at runtime.

NOTE: This does not work on Mac OS. Due to https://github.com/micropython/micropython/issues/5500

NOTE: Requires `micropython` program to installed (MicroPython Unix port).

To build and run tests of dynamic native modules on host
```
make check
```

You should see each of the test functions in tests/ being ran,
and then a summary at the end with something like:

```
...
Passed: 17
Failed: 0
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

#### Running tests on device

NOTE: Assumes that the .mpy files have been built first (in dist/).

We use `mpremote mount`, which allows the device to access files on the PC/host filesystem. This means we do not have to copy the modules or files across.

```
mpremote mount . run tests/test_all.py
```

You should see each of the test functions in tests/ being ran,
and then a summary at the end with something like:

```
...
Passed: 17
Failed: 0
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

