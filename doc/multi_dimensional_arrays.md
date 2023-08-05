
# Interoperable multi-dimensional arrays for MicroPython

# Motivation

Multi-dimensional arrays are widespread in data-intensive uses of MicroPython.
This includes applications such as Graphical User Interfaces, Image Processing / Computer Vision, Signal Processing and Machine Learning.

As of August 2020, each module tends to implements their own way of representing multi-dimensional data.
This limits efficient interoperability between different modules.

This is

# Status
**ONLY AN IDEA**





## Requirements gathering

## Examples of multi-dimensional data

The following are common uses of multi-dimensional arrays.

- Grayscale image. 2d. (width, height). Or 3d with 1 channel. (width, height, 1)
- Color image. 3d. (width, height, channels). Where channels=3 typ, for RGB/HSV/etc
- Video. 4d (time, width, height, channels).
- IMU time-series. 2d. (time, channels). Where channels=3,6,9 are common
- Spectrograms. 2d (time, )

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

### MAYBE

Read and write data at a particular index
Iterate forward item-wise over an axis

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

No direct support for interleaving

No direct support for packing booleans into uints.

This also means that there are no language-level convenience provided directly.
No iterators. No operator overloads. No slicing.

No direct support for broadcasting. Or universal functions.

It is envisioned that this also can be provided by external libraries,
and due to zero-copy interoperability, can be done in an efficient manner.


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

These are the strides into 

Array should always laid be out in C order.
TODO: row-major??
TODO: ASCII art illustration for 2d case


MAYBE. Alternative to array.array would be to use a buffer / buffer protocol.
However this means that the item size must also.
And does not automatically give interoperability with functions taking array.array in the base case


## Questions

- What about the scalar case? shape=()


Conventions
    Images: (width, height)

Functions are OK to operate only on a particular dimensions.
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
- Would it be possible to

## Modules

#### ulab

```
https://micropython-ulab.readthedocs.io/en/latest/ulab-programming.html#the-ndarray-object

Dimensions supported decided at compile time. Using ULAB_MAX_DIMS.
From 1 to 4 dimensions.
Choice influences code size significantly.
Up to 100kB with 4 dims.

uint8/int8/uint16/int16/float
float depends on the definition of mp_float_t, either float32 or double/float64
! no support for (u)int32/64

ndarray_obj_t
Has a boolean member. For interpreting data as packed bools.
Important for efficient boolean masks, common in numpy style code

Stores len in additional to shape. Says length is product of shapes.
Maybe optimization.
Unclear if supporting buffers that are not itemsize*prod(shape)

Stores shape and strides.
Strides are used to support sparse arrays.

ITERATE_VECTOR illustrates the data layout in-memory.

np.frombuffer(buffer, dtype=np.uint8)
Takes a 1d buffer as input.
Could be reshaped to interprete input as multidimensional
Does copy the data

ndarray_new_ndarray_from_tuple
creates, and allocates

??? could one create a new function that would use mdarray memory directly
Would allow operations that do not mutate size to work
And be a proof-of-concept that probably shape could also be migrated.
Or even doing a synced writeback to original

ndarray_new_foreign(data, dtype, shape, change_callback);
Where change_callback is a function that will be called when the ndarray has been modified.
Allows writing back the metadata
Calls to this would need to be inserted everywhere to end of public functions.
Alternatively one could sync-back manually using a utility function
A hack / proof-of-concept for the integration would be to monkey-patch the Python functions to call the sync
```


#### framebuf

https://docs.micropython.org/en/latest/library/framebuf.html

```
framebuf.FrameBuffer(buffer, width, height, format, stride)
```

Uses `buffer protocol` for the data buffer.

`format` specifies the type of pixel used in the FrameBuffer; permissible values are listed under Constants below. These set the number of bits used to encode a color value and the layout of these bits in buffer. Where a color value c is passed to a method, c is a small integer with an encoding that is dependent on the format of the FrameBuffer.

Supports `stride`. Number of pixels between each horizontal line of pixels in the FrameBuffer

Data is stored in ? format

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

