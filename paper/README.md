
# emlearn-micropython JOSS paper

## Building PDF from paper

How to build the paper locally. Requires Docker.

```
docker run --rm --volume $PWD/paper:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
```

## Runnable example

There is a plot in the paper,
that demonstrates using emlearn_fft and emlearn_trees from emlearn-micropython
to do Human Activity Recognition.

The MicroPython code is found in:

- [example.py](example.py)

The following Python files are used for support:

- [create_plot.py](create_plot.py)
- [processing.py](processing.py)
- [plotting.py](plotting.py)
- [extract_data.py](extract_data.py)

### Prerequities

- Python 3.10 or later
- [MicroPython, Unix port](https://github.com/micropython/micropython/tree/master/ports/unix).

### Install Python dependencies

```
pip install plotly numpy pandas scikit-learn emlearn setuptools pyarrow
```

### Install MicroPython modules

```
micropython -m mip install github:jonnor/micropython-npyfile
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_trees.mpy
micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_fft.mpy
```

### Run it

```
python create_plot.py
```

Should update the file `physical_activity_recognition_pipeline.png`.
