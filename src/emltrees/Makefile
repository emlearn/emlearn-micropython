# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# The ABI version for .mpy files
MPY_ABI_VERSION := 6.2

# Location of emlearn library
EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")


DIST_DIR := ../../dist/$(ARCH)_$(MPY_ABI_VERSION)

# Name of module
# NOTE: cannot contain _ - functions do not register on the module then?
MOD = emltrees

# Source files (.c or .py)
SRC = trees.c trees.py

# Stuff to make soft-float work
# If symbols are undefined, use tools/find_symbols.py to locate object files to add
SOFTFP_O := _arm_cmpsf2.o lesf2.o _arm_fixsfsi.o fixsfsi.o eqsf2.o gesf2.o _arm_addsubsf3.o _arm_muldivsf3.o addsf3.o _clzsi2.o _udivsi3.o floatsisf.o divsf3.o _thumb1_case_uqi.o
XTENSA_SOFTFP_O := _divsf3.o
SOFTFP_ENABLE := 0
ifeq ($(ARCH), armv6m)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7m)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), xtensawin)
	SRC_O += $(XTENSA_SOFTFP_O)
	CLEAN_EXTRA += $(XTENSA_SOFTFP_O)
endif

ifeq ($(SOFTFP_ENABLE), 1)
	SRC_O += $(SOFTFP_O)
	CLEAN_EXTRA += $(SOFTFP_O)
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

_arm_cmpsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

_divsf3.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(XTENSA_SOFTFP_O)

CFLAGS += -I$(EMLEARN_DIR)

dist: $(DIST_FILE)
