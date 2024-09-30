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
MOD = eml_iir_q15

# Source files (.c or .py)
SRC = iir_filter.c iir_filter.py

SRC += \
    ${CMSIS_DSP_DIR}/Source/FilteringFunctions/arm_biquad_cascade_df1_q15.c \
    ${CMSIS_DSP_DIR}/Source/FilteringFunctions/arm_biquad_cascade_df1_init_q15.c

# Stuff to make soft-float work
# If symbols are undefined, use tools/find_symbols.py to locate object files to add
SOFTFP_O := 


SOFTFP_ENABLE := 0
ifeq ($(ARCH), xtensawin)
    SOFTFP_O += _ashrdi3.o
	SOFTFP_ENABLE=1
endif

ifeq ($(ARCH), armv6m)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7m)
	SOFTFP_ENABLE=1
endif

ifeq ($(SOFTFP_ENABLE), 1)
	SRC_O += $(SOFTFP_O)
	#CLEAN_EXTRA += $(SOFTFP_O)
endif

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

_ashrdi3.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

# Set compile flag for non-ARM
# TODO: avoid this when target is actually arm
CFLAGS += -D__GNUC_PYTHON__

dist: $(DIST_FILE)
