
.. Places parent toc into the sidebar

:parenttoc: True

.. _getting_started_host:

=========================
Getting started on PC (Linux/MacOS/Windows)
=========================

.. currentmodule:: emlearn-micropython

emlearn-micropython runs on most platforms that MicroPython does.
This includes common desktop platforms such as Linux, Mac OS, Windows, et.c.
Since you need such a host platform to develop the Python machine-learning,
it is convenient also to do the first tests of the model on the host.


Prerequisites
===========================

You need to have installed **Python** (version 3.10+),
and a **C99 compiler** for your platform (GCC/Clang/MSVC).

On Windows, the **Windows Subsystem for Linux (WSL)** is recommended.

Install scikit-learn 
===========================

In this example, **scikit-learn** is used to train the models.

.. code-block:: console

    pip install scikit-learn

Install emlearn
===========================

The **emlearn** Python package will be used to convert the scikit-learn models
to something that can be loaded with emlearn-micropython.

.. code-block:: console

    pip install emlearn


Install MicroPython Unix port
==================================

We need to have the ``micropython`` interpreter installed.

Install the Unix port of MicroPython by following the `unix port documentation <https://github.com/micropython/micropython/tree/master/ports/unix#micropython-unix-port>`_


Install emlearn-micropython modules
====================================

emlearn-micropython is distributed as a set of MicroPython native modules.
These are .mpy file with native code, that can be installed at runtime using **mip**.
This example uses the ``emlearn_trees`` module, so that is what we will install.

.. code-block:: console

    micropython -m mip install https://emlearn.github.io/emlearn-micropython/builds/latest/x64_6.3/emlearn_trees.mpy

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

Run the program using `micropython`:

.. code-block:: console

    micropython xor_host.py


The output should be something like:

.. code-block:: console

    [0, 0] -> [1.0, 0.0] : False
    [32767, 32767] -> [0.666, 0.333] : False
    [0, 32767] -> [0.0, 1.0] : True
    [32767, 0] -> [0.0, 1.0] : True


Next
========

Now you have the emlearn-micropython running on your PC.
You may be interested in trying it out on a hardware device.
See for example :doc:`getting_started_device`.

