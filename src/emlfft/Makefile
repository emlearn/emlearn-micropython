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
MOD = emlfft

# Source files (.c or .py)
SRC = fft.c

LIBM_O := s_sincosf-fma.o
# math_err.o math_errf.o  
# s_sincosf.o
#SRC_O += $(LIBM_O)

# Stuff to make soft-float work
# If symbols are undefined, use tools/find_symbols.py to locate object files to add
SOFTFP_O := _arm_cmpsf2.o lesf2.o _arm_fixsfsi.o fixsfsi.o eqsf2.o gesf2.o addsf3.o mulsf3.o subsf3.o _clzsi2.o divdf3.o
SOFTFP_O += _dvmd_tls.o _aeabi_uldivmod.o _aeabi_ldivmod.o _divsi3.o _udivsi3.o _modsi3.o _umodsi3.o _thumb1_case_uhi.o  _udivmoddi4.o bpabi.o _clzdi2.o _ashldi3.o _lshrdi3.o _muldi3.o _divdi3.o _arm_muldivsf3.o _arm_addsubsf3.o
SOFTFP_ENABLE := 0
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

LIBM_FILENAME = /usr/lib/libm-2.39.a

#s_sincosf-fma.o:
#	$(CROSS)ar -x $(LIBM_FILENAME) $(LIBM_O)

_arm_cmpsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

CFLAGS += -I$(EMLEARN_DIR)

dist: $(DIST_FILE)
