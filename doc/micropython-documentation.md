
## Background
At the moment the MicroPython documentation is mostly missing a *user-guide*.
The existing documentation is mostly a *reference* (to ports/internals/tools).
A user in this context is an application/firmware developer:
Someone who wants to develop programs for a particular usecase,
and less interested in MicroPython internals etc.

Here are some things I would like to see, that helps people in more quickly learning how to build with MicroPython.

## User guide

- Getting started. zero to hero, simplified flow (mentioned earlier).
- Structuring programs
Project layout. Program flow, asyncio/blocking/nonblocking/interrupts.
- Development tools.
mpremote/commandline-based. IDEs and IDE extensions/plugins.
- Debugging MicroPython programs.
Tools, techniques. logging
- Selecting hardware
Limited scope: different capabilities of ports
- Using external libraries
Where to find. micropython-lib. Awesome MicroPython. How to install.
- Developing libraries
Project layout. Publishing.
- Automated testing
Host-based. On-device. Simulation with qemu. Hardware-in-the-loop
- Type checking.
stubs
- Continious integration/deployment
Building and publishing applications using CI


## More learning resources
Pointers to other places to learn about developing with MicroPython.
