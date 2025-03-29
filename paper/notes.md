

https://joss.readthedocs.io/en/latest/submitting.html

# TODO

Critical

- Write Statement of need
- Create a usage example. Including a nice plot

Non-critical

- Add a documentation page about the wider Data Science ecosystem?
- Tag newer version of emlearn-micropython ?
- Donate another 100 USD to JOSS

# Process

+ Write the paper
+ Submit
+ Respond to review comments


# Paper

WIP in joss-paper branch of emlearn-micropython


## Examples
https://joss.readthedocs.io/en/latest/example_paper.html

## Outline
Scope. Max 2 pages plus references

- Summary. 1-3 paragraphs. Max 1/2 page
- Statement of need. Up to 1/2 page
- Package contents. Table of the modules?
- Usage example. Short but illustrative. One attractive plot
- References

## What they ask for


    A list of the authors of the software and their affiliations, using the correct format (see the example below).

    A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

    A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.

    A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.

    Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.

    Acknowledgement of any financial support.
s

## Focus
+ Make it clear that it is relevant *for research* (eg in TinyML applications)
+ Make it likely that it will be cited.

# Related softwares

Things to cite

emlearn, TinyMaix
scikit-learn, keras, tensorflow / tf lite micro. numpy? scipy?

Alternatives
ulab, OpenMV.
Generating Python code. Using m2cgen, etc

For the implemented methods, the original papers

# Suggesting substantial scholarly effort

makes addressing research challenges significantly better (e.g., faster, easier, simpler).

emlearn-micropython makes research in Machine Learning for embedded systems easier.
This can both be applied research, and application oriented. Data collection and prototyping
Along with research in methods. By providing an example approach for developing ML methods for deployment on microcontrollers with MicroPython


# Suggesting citability
Already have one citation, 
https://www.sciencedirect.com/science/article/pii/S2352711024001493 

These papers identify a need for MicroPython to improve performance on numeric workloads
https://www.mdpi.com/2079-9292/12/1/143 
https://ieeexplore.ieee.org/abstract/document/9292199/
https://link.springer.com/chapter/10.1007/978-3-030-43364-2_4 

Real-Time Human Activity Recognition on Embedded Equipment: A Comparative Study
The C implementation on ESP32 still has a shorter processing time (0.0022 s) compared to MicroPython on ESP32 (0.15 s).


OpenMV
https://arxiv.org/abs/1711.10464

Example research targeting TinyML+MicroPython 
Could benefit from emlearn-micropython
https://www.mdpi.com/1424-8220/23/4/2344  (used ulab)
https://ieeexplore.ieee.org/abstract/document/8656727 
https://elifesciences.org/articles/67846 
https://www.sciencedirect.com/science/article/pii/S2772375523000138 



