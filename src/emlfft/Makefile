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

ifeq ($(ARCH), xtensawin)
    LIBM_O := lib_a-sf_cos.o lib_a-sf_sin.o lib_a-ef_rem_pio2.o lib_a-kf_cos.o lib_a-kf_sin.o lib_a-sf_fabs.o lib_a-kf_rem_pio2.o lib_a-sf_scalbn.o lib_a-sf_floor.o
endif

ifeq ($(ARCH), armv7emsp)
    LIBM_O := libm_a-sf_cos.o libm_a-sf_sin.o libm_a-ef_rem_pio2.o libm_a-kf_cos.o libm_a-kf_sin.o libm_a-sf_fabs.o libm_a-kf_rem_pio2.o libm_a-sf_scalbn.o libm_a-sf_floor.o
endif


# Stuff to make soft-float work
# If symbols are undefined, use tools/find_symbols.py to locate object files to add
ifeq ($(ARCH), armv7emsp)
	SOFTFP_O := _arm_addsubdf3.o _arm_muldivdf3.o _arm_muldivdf3.o _arm_truncdfsf2.o
endif


ifeq ($(ARCH), xtensawin)
	SOFTFP_O := _floatsidf.o _muldf3.o _divdf3.o _truncdfsf2.o
endif

SOFTFP_ENABLE := 0
ifeq ($(ARCH), armv6m)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7m)
	SOFTFP_ENABLE=1
endif

ifeq ($(ARCH), xtensawin)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7emsp)
	SOFTFP_ENABLE=1
endif

ifeq ($(SOFTFP_ENABLE), 1)
	SRC_O += $(SOFTFP_O)
	SRC_O += $(LIBM_O)
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

# FIXME: unhardcode these libm
ifeq ($(ARCH), xtensawin)
LIBM_FILENAME = /home/jon/.espressif/tools/xtensa-esp32-elf/esp-2022r1-11.2.0/xtensa-esp32-elf/xtensa-esp32-elf/lib/esp32-psram/no-rtti/libm.a
endif

ifeq ($(ARCH), armv7emsp)
	LIBM_FILENAME = /usr/arm-none-eabi/lib/thumb/v7e-m+fp/hard/libm.a
endif

libm_a-sf_cos.o:
	$(CROSS)ar -x $(LIBM_FILENAME) $(LIBM_O)

lib_a-sf_cos.o:
	$(CROSS)ar -x $(LIBM_FILENAME) $(LIBM_O)

_arm_truncdfsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

_truncdfsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)


CFLAGS += -I$(EMLEARN_DIR)

dist: $(DIST_FILE)
