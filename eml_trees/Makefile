# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

# Name of module
# NOTE: cannot contain _ - functions do not register on the module then?
MOD = emltrees

# Source files (.c or .py)
SRC = trees.c

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# Include to get the rules for compiling and linking the module
include $(MPY_DIR)/py/dynruntime.mk

CFLAGS += -I$(EMLEARN_DIR)
