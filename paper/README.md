
# emlearn-micropython JOSS paper

## Building PDF from paper

How to build the paper locally. Requires Docker.

```
docker run --rm --volume $PWD/paper:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
```

## Runnable example

There is a plot in the paper,
that demonstrates using `emlearn_fft` and `emlearn_trees` modules from emlearn-micropython
to do Human Activity Recognition.

The MicroPython code is found in:

- [example.py](example.py)

The following Python files are used for support:

- [create_plot.py](create_plot.py) - Top-level script/entrypoint
- [processing.py](processing.py) - Running MicroPython in subprocess on sensor data
- [plotting.py](plotting.py) - Plotting of data using Plotly
- [extract_data.py](extract_data.py) - Extracting example data from PAMAP2 dataset

### Prerequities

- Linux, Mac OS or Windows Subsystem for Linux (WSL)
- Python 3.10 or later
- [MicroPython, Unix port](https://github.com/micropython/micropython/tree/master/ports/unix).

### Install Python dependencies

```
pip install plotly numpy pandas scikit-learn emlearn setuptools pyarrow kaleido
```

### Install MicroPython modules


### Runing

#### Running using native modules

NOTE: not supported on Mac OS (due to https://github.com/micropython/micropython/issues/5500). See other method

Install the modules
```
micropython -m mip install github:jonnor/micropython-npyfile
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/latest/x64_6.3/emlearn_trees.mpy
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/latest/x64_6.3/emlearn_fft.mpy
```

Run the script
```
python create_plot.py
```

Should update the file `physical_activity_recognition_pipeline.png`.


#### Running using custom micropython executable

The modules in emlearn-micropython can be bundled into the `micropython` executable.
This is the only method that works on Mac OS at the moment.

Alternative 1: Download prebuilt binary (Mac OS)
```
curl -o micropython https://emlearn.github.io/emlearn-micropython/builds/latest/ports/macos/micropython
```
Alternative 1: Download prebuilt binary (Linux)
```
curl -o micropython https://emlearn.github.io/emlearn-micropython/builds/latest/ports/linux/micropython
```

Alternative 2: Use locally built micropython Unix port. Must first be built per [developing.md](https://github.com/emlearn/emlearn-micropython/blob/master/docs/developing.md#run-tests-on-pc-using-external-modules-build).
```
cp ../dist/ports/*/micropython ./
```


Install modules and run script
```
export PATH=./:$PATH
echo which micropython
curl -o npyfile.py https://raw.githubusercontent.com/jonnor/micropython-npyfile/refs/heads/master/npyfile.py
python create_plot.py
```

Should update the file `physical_activity_recognition_pipeline.png`.


