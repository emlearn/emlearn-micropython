
.. Places parent toc into the sidebar
:parenttoc: True

.. _external_modules:

===============================
External modules
===============================

An alternative, much simpler, installation method is as :ref:`native_modules` (recommended).

This method is more complicated, as it requires building MicroPython from scratch.
Therefore it is only recommended for 1) platforms which do not support native modules,
and 2) advanced users.

Supported versions
===========================

For general information about supported MicroPython versions, see :ref:`support`.
In principle, external modules should work on any hardware/port of MicroPython.


Prerequisites
===========================

External modules are included into the build process of MicroPython itself.
That means that you must have a MicroPython build environment set up,
including neccesary build toolchains et.c.
This process is specific for each MicroPython port.
Refer to the relevant port/X/README.md in the `micropython git repository <https://github.com/micropython/micropython/>`_ for the setup.

Always make sure that you can succesfully build and run the vanilla firmware
(no external modules), before you add in emlearn-micropython.

You need a git checkout of emlearn-micropython.
For example as a git submodule in your project.

::
    git clone https://github.com/emlearn/emlearn-micropython.git


Include external modules in build
===================================

emlearn-micropython contains both C modules, as well as Python modules.
Therefore it is neccesary to use both the options *USER_C_MODULES* and *FROZEN_MANIFEST*.

For details on this process, see official MicroPython documentation on
`building with external modules <https://docs.micropython.org/en/latest/develop/cmodules.html>`_.


Example build
----------------

For platforms that use a cmake-based build system.

::
	make USER_C_MODULES=./emlearn-micropython/src/micropython.cmake \
        FROZEN_MANIFEST=./emlearn-micropython/src/manifest.py \
        CFLAGS_EXTRA='-Wno-unused-function -Wno-unused-function'


For platforms that use a Makefile-based build.

::

	make USER_C_MODULES=./emlearn-micropython/src/ \
        FROZEN_MANIFEST=./emlearn-micropython/src/manifest.py \
        CFLAGS_EXTRA='-Wno-unused-function -Wno-unused-function'


Combining multiple modules
---------------------------

If you want to add more C modules and/or frozen manifest,
you will need to create your own `manifest.py` and `micropython.cmake` files.
Use the ones in emlearn-micropython as a starting point.


