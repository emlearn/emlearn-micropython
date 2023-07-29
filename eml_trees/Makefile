# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# Location of emlearn library
EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

# Name of module
# NOTE: cannot contain _ - functions do not register on the module then?
MOD = emltrees

# Source files (.c or .py)
SRC = trees.c

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
endif

$(info $(SOFTFP_ENABLE) $(SRC_O))

# None of these seem to work...
#CFLAGS += -static-libgcc
#CFLAGS += -mfloat-abi=soft
# CFLAGS += /usr/lib/gcc/arm-none-eabi/13.1.0/thumb/v6-m/nofp/libgcc.a
# arm-none-eabi-gcc: warning: /usr/lib/gcc/arm-none-eabi/13.1.0/thumb/v6-m/nofp/libgcc.a: linker input file unused because linking not done


# Include to get the rules for compiling and linking the module
include $(MPY_DIR)/py/dynruntime.mk

# CROSS is defined by the included
LIBGCC_FILENAME = $(shell $(CROSS)gcc $(CFLAGS) -print-libgcc-file-name)
$(info $(LIBGCC_FILENAME))

_arm_cmpsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

CFLAGS += -I$(EMLEARN_DIR)

