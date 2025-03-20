
.. Places parent toc into the sidebar

:parenttoc: True

.. _getting_started_host:

=========================
Getting started on PC (Linux/MacOS/Windows)
=========================

.. currentmodule:: emlearn-micropython

emlearn-micropython can run on PC using the Unix port of MicroPython.
This can be a very practical way to get started,
and in many cases a lot of the development can be done on the PC.

Prerequisites
===========================

You need to have installed **Python** (version 3.10+),
as well as MicroPython *Unix* port (version 1.24+).

On Windows, the Windows Subsystem for Linux (WSL) is recommended.
Alternatively you can use Docker for Windows.

Run MicroPython Unix port using Docker
===========================

.. code-block:: console

    docker run -ti --rm micropython/unix:v1.24.1 /bin/bash

Alternatively, you can build MicroPython for Unix yourself, by following the instructions in ports/unix
https://github.com/micropython/micropython/tree/master/ports/unix


Install emlearn-micropython modules
===========================

emlearn-micropython is distributed as a set of independent native modules.
Each module is a .mpy file, and can be installed with *mip*.

.. code-block:: console

    micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/master/x64_6.3/emlearn_trees.mpy
    
===========================


Next
========

Now you have the emlearn-micropython running on your PC.
You may be interested in trying it out on a hardware device.
See for example :doc:`getting_started_device`.

