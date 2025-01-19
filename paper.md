
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

- Summary. 1-3 paragraphs. Max 1/2 page
- Statement of need. Up to 1/2 page
- Package contents. Table of the modules?
- Usage example. Short but illustrative. One attractive plot

# Summary

```
1-3 paragraphs. Max 1/2 page
```


# Statement of need

```
Up to 1/2 page
```

[@scikit_learn]
[@keras]
[@tensorflow]

scipy.signal iir
[@scipy]


[@TinyMaix]
[@micropython]

[@tflite_micro] 

[@OpenMV]



Use in
research. Application oriented
educational

emlearn-micropython makes research in Machine Learning for embedded systems easier.
This can both be applied research, and application oriented. Data collection and prototyping
Along with research in methods. By providing an example approach for developing ML methods for deployment on microcontrollers with MicroPython


[ulab]


Generating Python code. Using
[m2cgen]


@misc{micropython,
  author = {George, Damien P and MicroPython contributors},
  title = {MicroPython - Python for Microcontrollers},
  year = {2014},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/micropython/micropython}}
}



# Package contents


| Module             | Typing          | Garbage Collected | Evaluation | Created |
|--------------------|:---------------:|:-----------------:|------------|---------|
| emlearn_cnn        | static, strong  | yes               | non-strict | 1990    |
| Lua                | dynamic, strong | yes               | strict     | 1993    |
| C                  | static, weak    | no                | strict     | 1972    |


# Usage example

```
Short but illustrative. One attractive plot
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

