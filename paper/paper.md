
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
```


# Statement of need

```
Up to 1/2 page
```

[@scikit-learn]
[@keras]
[@tensorflow]

scipy.signal iir
[@scipy]


[@micropython]

[@tflite_micro] 

[@OpenMV]

[@karavaev2024tinydecisiontreeclassifier]

[@emlearn]

Use in
research. Application oriented
educational

emlearn-micropython makes research in Machine Learning for embedded systems easier.
This can both be applied research, and application oriented. Data collection and prototyping
Along with research in methods. By providing an example approach for developing ML methods for deployment on microcontrollers with MicroPython


[@TinyMaix]
[@ulab]


Generating Python code. Using
[@m2cgen]


# Package contents


| Module             | Description |
|--------------------|:---------------:|
| emlearn_trees      | Decision tree ensembles | 
| emlearn_neighbors  | Nearest Neighbors    | 
| emlearn_cnn        | Convolutional Neural Network  | 
| emlearn_iir        | Infinite Impulse Response filters | 
| emlearn_fft        | Fast Fourier Transform  | 


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
and Jeremy Meyer for user-testing the CNN sub-module.


# References

