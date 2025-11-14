
.. Places parent toc into the sidebar
:parenttoc: True

.. _support:

===============================
Supported versions
===============================


Supported MicroPython versions
==============================

At any given point in time, emlearn-micropython only supports one MicroPython version.
In general we strongly encourage people to use the latest version.
There are no long-term-support or bugfix versions of emlearn-micropython, at this point.

==================  ======================
MicroPython         emlearn-micropython
==================  ======================
1.26.x              master
1.26.x              0.9.x
1.25.x              0.8.x
1.24.x              0.7.0
1.23.x              0.6.0
==================  ======================

If you build emlearn-micropython from source,
it might also work on a couple of MicroPython versions around the time of the commit, but this is not guaranteed.


Supported hardware
===========================

Due to limitations in MicroPython support for C modules,
the hardware support is a bit different for :ref:`native_modules` (recommended)
versus :ref:`external_modules` (advanced).
