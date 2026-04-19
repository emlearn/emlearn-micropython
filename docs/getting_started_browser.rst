
.. Places parent toc into the sidebar

:parenttoc: True

.. _getting_started_browser:

=========================
Getting started for browser
=========================

.. currentmodule:: emlearn-micropython

emlearn-micropython runs on most platforms that MicroPython does.
This includes running in a web browser, using the `Webassembly port <https://github.com/micropython/micropython/tree/master/ports/webassembly>`_ of MicroPython.
The browser integration is enabled by `PyScript <https://docs.pyscript.net/>`_.

Prerequisites
===========================

A web browser and a file editor.
An CPython 3.10+ installation is also recommended, to act as web server.


emlearn-micropython build for browser
==================================

We publish a pre-built MicroPython for each release.
This includes emlearn-micropython as external C modules.

There are two files needed, the `micropython.mjs` and `micropython.wasm`.
They can be downloaded from:

- https://raw.githubusercontent.com/emlearn/emlearn-micropython/refs/heads/gh-pages/builds/latest/ports/webassembly/micropython.mjs
- https://raw.githubusercontent.com/emlearn/emlearn-micropython/refs/heads/gh-pages/builds/latest/ports/webassembly/micropython.wasm

The ``latest`` version can be changed to a tag to have a specific version (``0.11.0`` or later).


Setup web page
==================================

Create an `index.html` page with the following contents:

.. literalinclude:: helloworld_browser/index.html
   :language: html

Make sure that you have ``micropython.mjs`` and ``micropython.wasm`` in the same directory.

Try it out 
========================

Start a HTTP server to serve the files

.. code-block:: console

    python -m http.server

Open your browser at http://localhost:8000

The MicroPython code using ``emlearn_linreg`` from emlearn-micropython should automatically run when you load the page.
The webpage should show an output like like:

``Input: [10.0, 75.0], prediction: 16.96 C``


Serving from device
====================================

On a MicroPython device with networking (like ESP32),
it can serve the browser frontend to clients.

We recommend using the excellent `MicroDot web framework <https://microdot.readthedocs.io/en/latest/>`_.

To be offline compatible, download PyScript and the MicroPython build files to your PC, and copy it to the device. Then update the HTML to have the local paths. See `PyScript offline <https://docs.pyscript.net/2026.3.1/user-guide/offline/#getting-micropython>`_ documentation for more information.



