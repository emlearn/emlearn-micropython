
# Sound level meter using Infinite Impulse Response (IIR) filters

This is a sound level meter implemented in MicroPython with
emlearn-micropython.
It implements the standard processing typically used in a
sound level meter used for noise measurements:
A frequency weighting and Fast (125ms) or Slow (1second)
time integration.
It then computes the soundlevel in decibels.
When using Fast integration, this measurement is known as LAF.
Or with Slow integration time, known as LAS.

## Hardware requirements

The device must have an `I2S` microphone,
and support the `machine.I2S` MicroPython module.
It has been tested on an ESP32 device, namely the LilyGo T-Camera Mic v1.6.

## Notes on measurement correctness

NOTE: In order to have reasonable output values,
the microphone sensitivity must be correctly specified.
Ideally you also check/calibrate wrt to a know good sound level meter.

NOTE: There is no compensation for non-linear frequency responses in microphone.

## Install requirements

Make sure to have Python 3.10+ installed.

Make sure to have the Unix port of MicroPython 1.23 setup.
On Windows you can use Windows Subsystem for Linux (WSL), or Docker.

Install the dependencies of this example:
```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running on host

```console
curl -o emliir.mpy https://github.com/emlearn/emlearn-micropython/raw/refs/heads/gh-pages/builds/master/x64_6.3/emliir.mpy
curl -o emlearn_arrayutils.mpy https://github.com/emlearn/emlearn-micropython/raw/refs/heads/gh-pages/builds/master/x64_6.3/emlearn_arrayutils.mpy

micropython soundlevel_test.py
```

## Running on device

!Make sure you have it running successfully on host first.

Flash your device with a standard MicroPython firmware,
from the MicroPython.org downloads page.

Download native modules.
```console

```

```console
mpremote cp device/emliir.mpy :
mpremote cp device/emlearn_arrayutils.mpy :
mpremote cp soundlevel.py :
mpremote run soundlevel_run.py
```

## Running with live camera input

This example requires hardware with SSD1306 screen,
in addition to an I2S microphone.
It has been tested on Lilygo T-Camera Mic v1.6.
By adapting the pins, it will probably also work on Lilygo T-Camera S3 v1.6.

```
mpremote run soundlevel_screen.py
```

