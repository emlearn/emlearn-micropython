
# Sound level meter

This is a sound level meter implemented in MicroPython with emlearn-micropython.
This can be used for basic sound measurements or noise monitoring over time.
It can also be used as a starting point for other applications that need to process sound.

The core code can be run on any MicroPyton port (including PC/Unix),
but to run the on-device examples you will need particular hardware (see hardware requirements section).

FIXME: image of the different examples. Show Device screen, Blynk dashboard

## Features

The [soundlevels.py](soundlevels.py) module implements the standard processing typically used in a
sound level meter used for noise measurements:

* Computes soundlevel in decibels SPL, using a digital microphone
* Supports *A* frequency weighting, or no weighting (*"Z"*)
* Supports *Fast* (125ms), *Slow* (1second) or *no* time integration.
* Supports showing the instantaneous sound level (short Leq)
* Supports summarized measurements: equivalent sound pressure level (Leq), minimum (Lmin), maximum (Lmax)
* Supports percentile-based summarizations: L10, L50, L80 etc (configurable)
* Supports 16 kHz sample rate

The following emlearn functionality are used in the implementation:

- `emliir`, IIR filters to implement the A weighting filter
- `emlearn_arrayutils.linear_map` to convert array values between `float` and `int16`


#### Notes on measurement correctness

NOTE: In order to have reasonable output values,
the *microphone sensitivity* **must be correctly specified**.
Ideally you also check/calibrate the output against a know good sound level meter.

NOTE: There is no compensation for non-linear frequency responses in microphone.

NOTE: processing is done largely in int16, which limits the dynamic range.

So the practical accuracy of your measurements will vary based on your hardware (microphone) and these limitations.
Generally this kind of device will be most useful to track relative changes,
and less useful for measuring against an absolute threshold limits (as specified in various noise regulation).
For known performance, get a sound level meter with a class rating (2/1), preferably with certification.

## Hardware requirements

The device must have an **I2S** digital microphone.
*PDM* microphone will not work, as they are not yet support in MicroPython (as of late 2024).

The microcontroller used must support the `machine.I2S` MicroPython module.
An ESP32 based device is recommended. But RP2, stm32, mimxrt also support I2S.
More information about I2S in MicroPython at [miketeachman/micropython-i2s-examples](https://github.com/miketeachman/micropython-i2s-examples).

Tested devices:

- LilyGo T-Camera Mic v1.6

Untested devices. Theoretically compatible by changing pinout

- [LilyGo T-Camera S3](https://www.lilygo.cc/products/t-camera-s3)
- Generic ESP32 with external I2S microphone breakout

Please report back if you get this running on a device not listed here.

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

## Example: Compute soundlevel for an audio file (on host/PC)

Install the emlearn modules
```console
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_arrayutils.mpy
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emliir.mpy

Compute soundlevels for a file
```console
micropython soundlevel_file.py test_burst.wav
```

## Example: Compute soundlevels on device

Recommended: Make sure you have it running successfully on host first.

Flash your device with a standard MicroPython firmware, from the MicroPython.org downloads page.

Download native modules.
```console

```

```console
mpremote cp device/emliir.mpy :
mpremote cp device/emlearn_arrayutils.mpy :
mpremote cp soundlevel.py :
mpremote run soundlevel_run.py
```

## Example: Log soundlevels to cloud over WiFi

This requires a device which has support for `network.WLAN` MicroPython module.
Typically an ESP32 or RP2.

Uses [Blynk](https://blynk.io/).




## Example: Show soundlevels on screen

This example requires hardware with **SSD1306 display**,
in addition to an I2S microphone.
It uses [micropython-nano-gui](https://github.com/peterhinch/micropython-nano-gui) for the screen.
So the example can be adapted to other displays supported by that framework.


```
mpremote run soundlevel_screen.py
```


## Development

### Running automated tests

Run automated tests
```console
python test_soundlevel.py
```

