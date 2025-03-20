# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# The ABI version for .mpy files
MPY_ABI_VERSION := 6.3

# Location of CMSIS-DSP library
CMSIS_DSP_DIR := ../../dependencies/CMSIS-DSP

DIST_DIR := ../../dist/$(ARCH)_$(MPY_ABI_VERSION)

# Name of module
MOD = emlearn_iir_q15

# Source files (.c or .py)
SRC = iir_filter.c iir_filter.py

SRC += \
    ${CMSIS_DSP_DIR}/Source/FilteringFunctions/arm_biquad_cascade_df1_q15.c \
    ${CMSIS_DSP_DIR}/Source/FilteringFunctions/arm_biquad_cascade_df1_init_q15.c

# enable linking of libm etc
LINK_RUNTIME=1

# Releases
DIST_FILE = $(DIST_DIR)/$(MOD).mpy
$(DIST_DIR):
	mkdir -p $@

$(DIST_FILE): $(MOD).mpy $(DIST_DIR)
	cp $< $@

# Include to get the rules for compiling and linking the module
include $(MPY_DIR)/py/dynruntime.mk

# CROSS is defined by the included
LIBGCC_FILENAME = $(shell $(CROSS)gcc $(CFLAGS) -print-libgcc-file-name)
$(info $(LIBGCC_FILENAME))

CFLAGS += -I$(CMSIS_DSP_DIR)/Include

$(CMSIS_DSP_DIR)/iir_q15.patched:
	cd $(CMSIS_DSP_DIR) && git apply -v ../df1_q15_disable_dsp.patch
	echo DONE > $@

# Set compile flag for non-ARM
# TODO: avoid this when target is actually arm
CFLAGS += -D__GNUC_PYTHON__

dist: $(CMSIS_DSP_DIR)/iir_q15.patched $(DIST_FILE)
