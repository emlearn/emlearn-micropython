
# XOR using Random Forest

The XOR problem stems from the famous Perceptrons (1969) book
by Marvin Minsky and Seymour Papert.
It can be trivially solved by a decision-tree or decision-tree ensemble.
Thus it is a kind of "Hello World" example.
Simple and an OK sanity check, but particularly useful.

## Install requirements

Make sure to have Python 3.10+ installed.

Make sure to have the Unix port of MicroPython 1.23 setup.
On Windows you can use Windows Subsystem for Linux (WSL), or Docker.

```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run training

This will train a RandomForest model using scikit-learn, and output `xor_model.csv`

```console
python xor_train.py
```

## Running on host

```console
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_trees.mpy

micropython xor_run.py
```

## Running on device (Viper IDE)

The fastest and easiest to to install on your device is to use Viper IDE.
This will install the library and the example code:
[<img src="https://raw.githubusercontent.com/vshymanskyy/ViperIDE/refs/heads/main/assets/btn_run.png" alt="Run using ViperIDE" height="42"/>](https://viper-ide.org/?install=github:emlearn/emlearn-micropython/examples/xor_trees)


## Running on device (manually)

!Make sure you have it running successfully on host first.

This command is for ESP32 (xtensawin).
For other hardware, replace the string.

```console
mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/master/xtensawin_6.3/emlearn_trees.mpy
mpremote cp xor_model.csv :
```

```console
mpremote run xor_run.py
```
