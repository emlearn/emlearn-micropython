
# Interoperable multi-dimensional arrays for MicroPython

# Motivation

Multi-dimensional arrays are widespread in data-intensive uses of MicroPython.
This includes applications such as Graphical User Interfaces, Image Processing / Computer Vision, Signal Processing and Machine Learning.

As of August 2020, each module tends to implements their own way of representing multi-dimensional data.
This limits efficient interoperability between different modules.


# Status
**ONLY AN IDEA**.

- **NOT EVEN SURE ITS A GOOD IDEA**
- Requirements gathering incomplete
- Several key questions unresolved
- Design is only an initial sketch


## Requirements gathering

## Examples of multi-dimensional data

The following are common uses of multi-dimensional arrays.

- Grayscale image. 2d. (width, height). Or 3d with 1 channel. (width, height, 1)
- Color image. 3d. (width, height, channels). Where channels=3 typ, for RGB/HSV/etc
- Video. 4d (time, width, height, channels).
- IMU time-series. 2d. (time, channels). Where channels=3,6,9 are common
- Spectrograms. 2d (time, bands).

NOTE: RGB images is often stored with R,G,B(,A) 8-bit packed into 32 bits

NOTE: 2-channel audio is often stored as 2xint16 packed into 32 bits

It is always possible to have multiple of such data items,
so it *might* be useful to support at least 1 more dimension.
But this could also be done with standard Python collections, such as lists.

### High-level requirements

The overarching goals are:

- Inter-operate between different modules.
Where a module is likely to implement some (but not all) data processing in C

In particular it must be possible to

- Construct a new array
- Modify values. Without copying
- Make array smaller. Without copying???
- Make array bigger. Neccitates allocation/copying
- Passing arrays between different functions
- Interoperate with **existing** modules/code
- Loading from disk
- Saving to disk

All these should be possible both in Python and in C,
and also working in user modules and native modules.


## Design

## Naming

Current terminology: *mdarray* - for multi-dimensional-array.
But maybe this is too similar visually and in terms of pronounciation to *ndarray*. 

Not using *ndarray*, since it is already in use by ulab,
as this is slight different take on the problem - and not direct replacement.

## Scope

This proposal is only about **data representation**,
in a way that enables basic interoperability between different modules.

Meaning that all *operations* on the data is outside of the scope,
and is up to individual modules to provides.

#### In scope

Standard 

Support at least 4 dimensional data.

#### Undecided

##### Item packing/interleaving
? what do do with packing of data inside array items
Examples: uint8 RGB(A) in its many permutations, int16 L/R audio, or boolean/bits packed into 32-bit items.

This can get arbitrarily complex. Like RGB565 and other non-uniform packing.
Could leave it all to the consumers/producers.
BUT, it means that the interleaving needs to be communicated out-of-channel,
which is inconvenient and fragile.

Is it possible to make an mechanism?
Either as a fully opaque thing.
Or a standardized+exensible option.
`item_format` which is a kind of enumeration?

#### Out of scope

All high-level operations are out of scope.

This also means that there are no language-level convenience provided directly.
No iterators. No operator overloads. No slicing.

No direct support for broadcasting. Or universal functions.

It is envisioned that this also can be provided by external libraries,
and due to zero-copy interoperability, can be done in an efficient manner.

No support for scalar value. shape=() in numpy.
Not sure if implemented in ulab even?

#### Rationale

TODO: once things shake out, write abit about why certain choices were made


## Class definition

```
class mdarray
    shape: tuple
    array: array.array
```

NOTE: In practice this should possibly be implemented in C inside MicroPython.

Where `array` is the 1d continious buffer which stores the data, as well as the item format.
And `shape` is a tuple of the dimensions of the array.

MAYBE. Alternative to array.array would be to use a buffer / buffer protocol.
However this means that the item size must also.
And does not automatically give interoperability with functions taking array.array in the base case

## Data layout

Array should always laid be out in C order.

The semantic ordering should be addressed with a set of conventions.
This should reflect the most ordering that is most common in practice,
such that copy-free operation is most likely (but not guaranteed) to be possible.

For example:

- Images: (width, height)
- Spectrogram: (time, bands)

### Built-in operations

Fundamental

- Create mdarray
- Pass between functions
- Access and modify the underlying `.array` with data/values 
- Access and modify the `.shape`

Maybe

- Read and write data at a particular index
- Iterate forward item-wise over specified axis

## Operations

Functions are OK to support operate only working with on a particular sizes or data-types.
Example: image functions that only work on 2d data
Should check the length.
that operate on 2d. Should throw when


# Existing usages of multi-dimensional arrays

An review of existing MicroPython modules that use multi-dimensional arrays.
With focus on understanding of how feasible it is to get interoperability,
and what particularities are important in designing a base for interoperability. 

## Key questions

- How is the data buffer/array stored? Anyone using `array.array`?
- How are the dimensions stored?
- What order is the data stored in?
- Would it be possible to operate on a `mdarray` without intrusive modifications?
- How could an implementation strategy look like?


## Modules


#### framebuf

https://docs.micropython.org/en/latest/library/framebuf.html

```
framebuf.FrameBuffer(buffer, width, height, format, stride)
```

Uses `buffer protocol` for the data buffer.

`format` is an enumeration for the item packing.
7 different values, from 1 bit to RGB565.

Supports `stride`. Number of pixels between each horizontal line of pixels in the FrameBuffer

Data is stored in `TODO ?` order


#### lvgl

TODO: research their implementation

#### ulab

```
typedef struct _ndarray_obj_t {
    mp_obj_base_t base;
    uint8_t dtype;
    uint8_t itemsize;
    uint8_t boolean;
    uint8_t ndim;
    size_t len;
    size_t shape[ULAB_MAX_DIMS];
    int32_t strides[ULAB_MAX_DIMS];
    void *array;
} ndarray_obj_t;
```
https://micropython-ulab.readthedocs.io/en/latest/ulab-programming.html#the-ndarray-object

Dimensions supported decided at compile time. Using ULAB_MAX_DIMS.
From 1 to 4 dimensions.

Item size can be `uint8/int8/uint16/int16/float`.
float depends on the definition of mp_float_t, either float32 or double/float64
! no support for (u)int32/64

ndarray_obj_t
Has a `boolean` member. For interpreting data as packed bools.
Important for efficient boolean masks, common in numpy style code

Stores `len` in additional to shape. Says length is product of shapes.
Maybe optimization.
Unclear if supporting buffers that are not itemsize*prod(shape). `TODO: figure out`?

Stores shape and strides.
Strides are used to support sparse arrays.

`ITERATE_VECTOR` illustrates the data layout in-memory.

```
np.frombuffer(buffer, dtype=np.uint8)
```

Takes a 1d buffer as input.
Could be reshaped to interprete input as multidimensional
Does copy the data

ndarray_new_ndarray_from_tuple
creates, and allocates

??? could one create a new function that would use mdarray memory directly
Would allow operations that do not mutate size to work
And be a proof-of-concept that probably shape could also be migrated.
Or even doing a synced writeback to original

```
ndarray_new_foreign(data, dtype, shape, change_callback);
```

Where change_callback is a function that will be called when the ndarray has been modified.
Allows writing back the metadata
Calls to this would need to be inserted everywhere to end of public functions.
Alternatively one could sync-back manually using a utility function
A hack / proof-of-concept for the integration would be to monkey-patch the Python functions to call the sync


#### OpenMV

Core multi-dimensional array: Image

https://github.com/openmv/openmv/blob/master/src/omv/imlib/imlib.h

```
typedef struct image {
    int32_t w;
    int32_t h;
    PIXFORMAT_STRUCT;
    union {
        uint8_t *pixels;
        uint8_t *data;
    };
} image_t;
```

https://github.com/openmv/openmv/blob/master/src/omv/modules/py_image.c

```
typedef struct _py_image_obj_t {
    mp_obj_base_t base;
    image_t _cobj;
} py_image_obj_t;
```

Could one counstruct this object, where it is a view onto a `mdarray` ?

https://docs.openmv.io/library/omv.image.html

https://docs.openmv.io/library/omv.tf.html
takes either img or array

https://docs.openmv.io/library/omv.imu.html
just returns tuples for accel/gyro data - no time-series



# References
 
### array.array

https://docs.micropython.org/en/latest/library/array.html
https://docs.python.org/3/library/array.html

# Misc

## Proof of concept

IO / serialization options

- Image files. .jpg / .png
- .npy files

