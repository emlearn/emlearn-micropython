# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# Location of emlearn library
EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

DIST_DIR := ../dist

# Name of module
# NOTE: cannot contain _ - functions do not register on the module then?
MOD = emlneighbors

# Source files (.c or .py)
SRC = neighbors.c

# Stuff to make soft-float work
SOFTFP_O := _arm_cmpsf2.o lesf2.o _arm_fixsfsi.o fixsfsi.o eqsf2.o gesf2.o
SOFTFP_ENABLE := 0
ifeq ($(ARCH), armv6m)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7m)
	SOFTFP_ENABLE=1
endif

ifeq ($(SOFTFP_ENABLE), 1)
	SRC_O += $(SOFTFP_O)
	CLEAN_EXTRA += $(SOFTFP_O)
endif

# for qsort
LIBC_O := msort.o
SRC_O += $(LIBC_O)
# XXX: does not work. Needs memcpy, and maybe other things also
# perhaps use https://github.com/svpv/qsort instead?

# Include to get the rules for compiling and linking the module
include $(MPY_DIR)/py/dynruntime.mk

# for qsort
SRC_O += $(LIBC_O)

# Releases
DIST_FILE = $(DIST_DIR)/$(MOD)_$(ARCH).mpy
$(DIST_DIR):
	mkdir -p $@

$(DIST_FILE): $(MOD).mpy $(DIST_DIR)
	cp $< $@

# CROSS is defined by the included
LIBGCC_FILENAME = $(shell $(CROSS)gcc $(CFLAGS) -print-libgcc-file-name)
$(info $(LIBGCC_FILENAME))

LIBC_FILENAME = $(shell $(CROSS)gcc $(CFLAGS) -print-file-name=libc.a)
$(info $(LIBC_FILENAME))

_arm_cmpsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

# msort contains qsort
msort.o: 
	$(CROSS)ar -x $(LIBC_FILENAME) $(LIBC_O)

CFLAGS += -I$(EMLEARN_DIR)

dist: $(DIST_FILE)
