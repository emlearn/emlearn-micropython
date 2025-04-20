
# Dynamic native modules are the future

WIP on a post for MicroPython Discussion forum,
advocating for dynamic native modules - their potential and how they can be further improved.

Dynamic modules are .mpy files with native code that can be installed at runtime.
This enables to "mip install" modules that extend the 

For computationally intensive uses C modules are very useful,
because they can be 10-100x faster than Python (even when using native and Viper emitters).
This includes things like data processing, signal processing, machine learning, compression, cryptography, and much more.

The facilities for dynamic native modules for several years already,
but it has not seen that much usage or testing.
However in MicroPython 1.24 (October 2024) critical issues where fixed,
and now in MicroPython 1.25 (April 2025) there is support for linking symbols,
which means that it is getting ready for wider usage.

## What are dynamic native modules

natmod, https://docs.micropython.org/en/latest/develop/natmod.html
extmod, https://docs.micropython.org/en/latest/develop/cmodules.html

## Advantages of native modules

User benefits, user-experience
Library developer benefits
Hardware manufacturer benefits

Enables having C module extensions without requiring custom builds

## Drawbacks of native modules

RAM usage.
Code dupliation.

## What to use native modules for (and not)



## Painpoints with native modules

TODO: import from other notes

These are all things which can be improved, and it is up to us all to help out.
I believe that this can significantly improve the MicroPython ecosystem.

## Building

TODO: file issues around these things, at least the most obvious/clear ones

Ideas

- Support for natmod in micropython-lib
- Distribute the natmods in micropython. btree
- mpremote. Resolve correct architecture for native .mpy files
- awesome-micropython. Include standard marker for native modules?
- mpbuild. Support for natmod building
- docs. Document better how to support both extmod and natmod with same codebase


## Call to Action

Library developers.
Consider supporting native modules for your C-based library modules.
Try to develop new modules as natmod-first, with extmod as the backup
(for ports without natmod or uses where ).

MicroPython contributors.
Help out to improve 

## Examples of native modules in use

Libraries

- [BrianPugh/tamp](https://github.com/BrianPugh/tamp) - a low-memory, DEFLATE-inspired lossless compression library.
- emlearn-micropython - Efficient and convenient Machine Learning engine
- **Your library here!!**

Tools

- ViperIDE. 
- **Your tools here!**


# Thanks to

vhymassky
agatti
damien
BrianPugh


