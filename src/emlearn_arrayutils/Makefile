# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# The ABI version for .mpy files
MPY_ABI_VERSION := 6.3

DIST_DIR := ../../dist/$(ARCH)_$(MPY_ABI_VERSION)

# Name of module
MOD = emlearn_arrayutils

# Source files (.c or .py)
SRC = modarrayutils.c

# Stuff to make soft-float work
# If symbols are undefined, use tools/find_symbols.py to locate object files to add
SOFTFP_O :=
SOFTFP_ENABLE := 0

ARM_SOFTFP_O := _arm_cmpsf2.o lesf2.o divsf3.o _thumb1_case_uqi.o _arm_fixsfsi.o floatsisf.o fixsfsi.o eqsf2.o gesf2.o addsf3.o mulsf3.o subsf3.o _clzsi2.o _udivsi3.o
ARM_SOFTFP_O += _arm_muldivsf3.o _arm_addsubsf3.o

ifeq ($(ARCH), xtensawin)
    SOFTFP_O += _divsf3.o _ashrdi3.o
	SOFTFP_ENABLE=1
endif

ifeq ($(ARCH), armv6m)
    SOFTFP_O += $(ARM_SOFTFP_O)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7m)
    SOFTFP_O += $(ARM_SOFTFP_O)
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

_arm_cmpsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

_divsf3.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)


dist: $(DIST_FILE)
