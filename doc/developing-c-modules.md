
# Developing C modules for MicroPython

This is a sketch for a presentation about developing
third-party libraries for MicroPython as C modules.

## Outline

- External C modules, what is it, how does it work.
- Native .mpy modules, what is it, how does it work.
Limitations. Fundamental and incidental/current.
- Supporting both as external C module and native.
Some #ifdefs and duplication needed. Potential for unified macros?
- CMake and Makefile setups for external C modules
- Writing automated tests.
Mostly like for .py files. pytest-style, use asserts with message
- Running automated tests against hardware.
Using mpremote mount. Modify sys.path for architecture specifics
- Continious Integration setup.
Using Github Actions. Can reuse ci shell scripts from MicroPython repo.
- Continious Delivery setup.
Using Github Actions -> Github pages. .mpy files
- Supporting browser using Webassembly.
Standard external C modules build.
- Supporting running on PC
! Native modules do not work in Mac OS.
! Windows only supported using Windows Subsystem for Linux?
- API refererence documentation.
Using stub files sphinx with sphinx-autodoc-typehints
- ?? How to use the stubs for typechecking, like micropython stubs

## Examples

- emlearn-micropython, https://github.com/emlearn/emlearn-micropython/
- tamp, https://github.com/BrianPugh/tamp/tree/main/mpy_bindings
- ulab, https://github.com/v923z/micropython-ulab
