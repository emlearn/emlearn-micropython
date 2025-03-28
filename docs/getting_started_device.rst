
.. Places parent toc into the sidebar

:parenttoc: True

.. _getting_started_device:

==================================================
Getting started on device (ESP32/RP2/STM32/etc)
==================================================

.. currentmodule:: emlearn-micropython

emlearn-micropython runs on most hardware platforms that MicroPython does.


Prerequisites
===========================

Ensure that you have **Python** and  **emlearn** setup as per the :doc:`getting_started_host`.

Ensure that you have MicroPython flashed onto your device.
Check the `Download section <https://micropython.org/download>`_ for MicroPython.

Install mpremote
===========================

``mpremote`` is used to run scripts and copy files to/from a hardware device running MicroPython.

.. code-block:: console

    pip install mpremote


Install emlearn-micropython modules
====================================

emlearn-micropython is distributed as a set of MicroPython native modules.
These are .npy file with native code, that can be installed at runtime using **mip**.
This example uses the ``emlearn_trees`` module, so that is what we will install.

For ESP32 use the ``xtensawin`` architecture.

.. code-block:: console

    mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/master/xtensawin_6.3/emlearn_trees.mpy

For ARM Cortex M4F/M33/M7 etc use the ``armv7emsp`` architecture.

.. code-block:: console

    mpremote mip install https://emlearn.github.io/emlearn-micropython/builds/master/armv7emsp_6.3/emlearn_trees.mpy

For more details about architectures for native modules, see `MicroPython mpyfiles documentation <https://docs.micropython.org/en/latest/reference/mpyfiles.html#versioning-and-compatibility-of-mpy-files>`_


Create model in Python
===========================

We will train a simple model to learn the XOR function.
The same steps will be used for model of any complexity.
Copy and save this as file ``xor_train.py``.

.. literalinclude:: helloworld_xor/xor_train.py
   :language: python
   :emphasize-lines: 3-4,26,30
   :linenos:

Run the script

.. code-block:: console

    python xor_train.py

It will generate a file ``xor_model.csv`` containing the C code for our model.


Use in MicroPython code 
========================

To run our model we use a simple MicroPython program.

Copy and save this as file ``xor_run.py``.

.. literalinclude:: helloworld_xor/xor_run.py
   :language: python
   :emphasize-lines: 3,10,23
   :linenos:


Try it out 
========================

In our training data input values above ``2**14`` is considered "true".
So for the XOR function, if **one and only one** of the values is above this limit, should get class **1** as output - else class **0**. 

Run the program on device using ``mpremote run``:

.. code-block:: console

    mpremote run xor_host.py


The output should be something like:

.. code-block:: console

    [0, 0] -> [1.0, 0.0] : False
    [32767, 32767] -> [0.666, 0.333] : False
    [0, 32767] -> [0.0, 1.0] : True
    [32767, 0] -> [0.0, 1.0] : True

Next
========

Now you have the emlearn-micropython on running hardware.
For information on practical usecases, see :doc:`examples`.
Otherwise check out :doc:`api_reference`.


