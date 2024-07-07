# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# The ABI version for .mpy files
MPY_ABI_VERSION := 6.2

# Location of deps
TINYMAIX_DIR := ../../dependencies/TinyMaix

DIST_DIR := ../../dist/$(ARCH)_$(MPY_ABI_VERSION)

# Name of module
MOD = tinymaix_cnn

# Source files (.c or .py)
SRC = mod_cnn.c

# Stuff to make soft-float work
# If symbols are undefined, use tools/find_symbols.py to locate object files to add
SOFTFP_O :=
SOFTFP_ENABLE := 0
ifeq ($(ARCH), armv6m)
	SOFTFP_ENABLE=1
#    SOFTFP_O += 
endif
ifeq ($(ARCH), armv7m)
	SOFTFP_ENABLE=1
    SOFTFP_O += _arm_addsubsf3.o _arm_muldivsf3.o _arm_fixsfsi.o _arm_cmpsf2.o _arm_fixunssfsi.o _arm_addsubdf3.o _arm_truncdfsf2.o
endif
ifeq ($(ARCH), armv7emsp)
	SOFTFP_ENABLE=1
    SOFTFP_O += _arm_addsubdf3.o _arm_truncdfsf2.o
endif
ifeq ($(ARCH), xtensawin)
	SOFTFP_ENABLE=1
    SOFTFP_O += _truncdfsf2.o _addsubdf3.o _extendsfdf2.o _divsf3.o
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

_addsubdf3.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

_arm_addsubdf3.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

CFLAGS += -I$(TINYMAIX_DIR)/include -I$(TINYMAIX_DIR)/src -Wno-error=unused-variable -Wno-error=multichar

dist: $(DIST_FILE)
