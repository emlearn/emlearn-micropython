
.. Places parent toc into the sidebar
:parenttoc: True

.. _native_modules:

=========================
Native modules
=========================

This is the recommended installation method on the majority of hardware platform,
especially when getting started.
An alternative, more complicated, installation method is documented in :ref:`external_modules`.


Supported versions
===========================

The correct .mpy files to use depends on the CPU architecture of your microcontroller, as well as the MicroPython version.
Information is also available in the official documentation: `MicroPython: .mpy files <https://docs.micropython.org/en/latest/reference/mpyfiles.html#versioning-and-compatibility-of-mpy-files>`_.

The following are **supported** as native modules at this time:
``x64``, ``armv7m``, ``arm6vm`, ``armv7emsp``, ``xtensawin``, ``rv32imc``.

The following are **not supported**: ``xtensa``, ``x86``.
For these you will instead need to use :ref:`external_modules`.
There are no plans to support ``xtensa`` (ESP8266) or ``x86`` as native modules,
as these are very old platforms.


Find .mpy ABI version
-------------------------

The following is an overview of .mpy ABI version for MicroPython releases.

+---------------------+---------------+
| MicroPython version | .mpy version  |
+=====================+===============+
| 1.26.x              | 6.3           |
+---------------------+---------------+
| 1.25.x              | 6.3           |
+---------------------+---------------+
| 1.24.x              | 6.3           |
+---------------------+---------------+
| 1.23.x              | 6.3           |
+---------------------+---------------+


Find .mpy architecture
-------------------------

Identify which CPU architecture your device uses.
You need to specify `ARCH` to install the correct module version.

+--------------+------------------------------------+------------------------+
| ARCH         | Description                        | Examples               |
+==============+====================================+========================+
| x64          | x86 64 bit                         | PC                     |
+--------------+------------------------------------+------------------------+
| x86          | x86 32 bit                         |                        |
+--------------+------------------------------------+------------------------+
| armv6m       | ARM Thumb (1)                      | Cortex-M0              |
+--------------+------------------------------------+------------------------+
| armv7m       | ARM Thumb 2                        | Cortex-M3              |
+--------------+------------------------------------+------------------------+
| armv7emsp    | ARM Thumb 2, single float          | Cortex-M4F, Cortex-M7  |
+--------------+------------------------------------+------------------------+
| armv7emdp    | ARM Thumb 2, double floats         | Cortex-M7              |
+--------------+------------------------------------+------------------------+
| xtensa       | non-windowed                       | ESP8266                |
+--------------+------------------------------------+------------------------+
| xtensawin    | windowed with window size 8        | ESP32, ESP32-S3        |
+--------------+------------------------------------+------------------------+
| rv32imc      | RISC-V                             | ESP32-C3, ESP32-C6     |
+--------------+------------------------------------+------------------------+



Prebuilt native modules
===========================

All the modules in emlearn-micropython for supported architectures are available pre-built.
You can `browse them on Github <https://github.com/emlearn/emlearn-micropython/tree/gh-pages/builds>`_.
And they are available over HTTPS.
The directory structure is as follows:

::

    https://emlearn.github.io/emlearn-micropython/builds/$VERSION/$ARCH_$ABI/$MODULE.mpy

    where:
    VERSION=master|0.9.0
    MODULE=emlearn_trees
    ARCH=xtensawin
    ABI=6.3


Installing using mip
===========================

Native modules can be installed using the `mip package manager <https://docs.micropython.org/en/latest/reference/packages.html>`_.


For example, to install ``emlearn_trees`` for MicroPython 1.25 (ABI 6.3) on ESP32 (xtensawin), use:

::

    mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/master/xtensawin_6.3/emlearn_trees.mpy

