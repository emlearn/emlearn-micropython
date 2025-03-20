# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# The ABI version for .mpy files
MPY_ABI_VERSION := 6.3

# Location of emlearn library
EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

# enable linking of libm etc
LINK_RUNTIME=1

DIST_DIR := ../../dist/$(ARCH)_$(MPY_ABI_VERSION)

# Name of module
MOD = emlearn_iir

# Source files (.c or .py)
SRC = iir_filter.c

# Releases
DIST_FILE = $(DIST_DIR)/$(MOD).mpy
$(DIST_DIR):
	mkdir -p $@

$(DIST_FILE): $(MOD).mpy $(DIST_DIR)
	cp $< $@

# Include to get the rules for compiling and linking the module
include $(MPY_DIR)/py/dynruntime.mk

CFLAGS += -I$(EMLEARN_DIR)

dist: $(DIST_FILE)
