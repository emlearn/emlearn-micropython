
---
title: 'emlearn-micropython: Machine Learning and Digital Signal Processing for MicroPython'
tags:
  - MicroPython
  - TinyML
  - Machine Learning
  - Microcontroller
  - Embedded Systems
authors:
  - name: Jon Nordby
    orcid: 0000-0002-0245-7456
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Soundsensing, Norway
   index: 1
date: 19 January 2025
bibliography: paper.bib

---

# Summary

```
1-3 paragraphs. Max 1/2 page

A summary describing the high-level functionality
and purpose of the software =
or a diverse, non-specialist audience.
```


# Statement of need

```
Up to 1/2 page
```

Running machine learning inference directly on embedded device
close to the sensor
Is being applied to a wide range of tasks
Enables low-cost
Over the last decade

Still research remaining
improve the practicality

Use in
research. Application oriented
educational

On the computer, Python is the most commonly used application language for machine learning.

It is also becoming feasible to use the Python language on microcontrollers,
thanks to implementations such as MicroPython[@micropython] which is tailor-made for such applications.

The library ulab[@ulab] implements efficient numeric computing facilities,
including the core parts of numpy, plus some parts of scipy.
However, as of 2025 there are no implementations of machine learning algorithms.

It is possible to generate Python code, using tools such as m2cgen[@m2cgen].

[@OpenMV]


emlearn-micropython makes research in Machine Learning for embedded systems easier.
This can both be applied research, and application oriented. Data collection and prototyping
Along with research in methods.
By providing an example approach for developing ML methods for deployment on microcontrollers with MicroPython

Can do high-level parts of algorithm in Python.
And then use optimization of critical sections as C modules.


[@scikit-learn]
[@keras]
[@tensorflow]

[@scipy]
[@numpy]

[@tflite_micro] 


[@karavaev2024tinydecisiontreeclassifier]

Generating Python code. Using


# Package contents

The emlearn-micropython software package provides a selection of machine learning inference algorithms,
along with some Digital Signal Processing functions.
They have been selected based on what is useful and commonly used in embedded systems.
Table \ref{identifier} provides an listing of the provided functionality.

The software is distributes as MicroPython native modules[@micropython_native_module].
A MicroPython native module contains a combination of machine code (compiled from C) and MicroPython interpret byte-code (compiled from Python).
The module can installed at runtime using the `mip` package manager.

The modules provided by emlearn-micropython are independent of eachother, and typically a few kilobytes large.
This makes it easy to install just what is needed for a particular application.


| Module             | Description                          | Corresponds to |
|:-------------------|:-------------------------------------|:----------------------------------|
| emlearn_trees      | Decision tree ensembles              | sklearn RandomForestClassifier    |
| emlearn_neighbors  | Nearest Neighbors                    | sklearn KNeighborsClassifier      |
| emlearn_cnn        | Convolutional Neural Network         | keras Model+Conv2D                |
| emlearn_fft        | Fast Fourier Transform               | scipy.fft.fft                     |
| emlearn_iir        | Infinite Impulse Response filters    | scipy.signal.sosfilt              |
| emlearn_arrayutils | Fast utilities for array.array       | N/A                               |
 
Table: Overview of modules provided by emlearn-micropython \label{modules}

Most of the modules are implemented as wrappers of functionality provided in the emlearn C library[@emlearn].
However, the emlearn_cnn module is implemented using the TinyMaix library[@TinyMaix].

# Usage example

```
Short but illustrative example
One attractive plot
```


Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We would like to thank
Volodymyr Shymanskyy for his work on improving native module support in MicroPython,
Damien P. George for fixes to native modules (and generally for MicroPython maintenance),
and Jeremy Meyer for user-testing of the emlearn_cnn sub-module.


# References

