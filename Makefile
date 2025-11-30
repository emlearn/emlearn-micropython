
ARCH ?= x64
MPY_ABI_VERSION ?= 6.3
MPY_DIR ?= ../micropython
MICROPYTHON_BIN ?= micropython

# extmod settings
PORT=unix
BOARD=ESP32_GENERIC_S3

VERSION := $(shell git describe --tags --always)

MPY_DIR_ABS = $(abspath $(MPY_DIR)) 

C_MODULES_SRC_PATH = $(abspath ./src)

ifeq ($(PORT),unix)
    MANIFEST_PATH=$(abspath ./src/manifest_unix.py)
else
    MANIFEST_PATH=$(abspath ./src/manifest.py)
endif


MODULES_PATH = ./dist/$(ARCH)_$(MPY_ABI_VERSION)
PORT_DIR = ./dist/ports/$(PORT)
PORT_BUILD_DIR=$(MPY_DIR)/ports/$(PORT)/build-$(BOARD)
PORT_DIST_DIR=./dist/ports/$(PORT)/$(BOARD)

UNIX_MICROPYTHON = ./dist/ports/unix/micropython


# List of modules
MODULES = emlearn_trees \
	emlearn_neighbors \
	emlearn_iir \
	emlearn_fft \
	emlearn_kmeans \
	emlearn_iir_q15 \
	emlearn_arrayutils \
	emlearn_linreg \
	emlearn_cnn_int8 \
	emlearn_cnn_fp32

# Generate list of .mpy files
MODULE_MPYS = $(addprefix $(MODULES_PATH)/,$(addsuffix .mpy,$(MODULES)))

# Special cases
emlearn_cnn_int8_SRC = src/tinymaix_cnn
emlearn_cnn_int8_CONFIG = CONFIG=int8
emlearn_cnn_fp32_SRC = src/tinymaix_cnn
emlearn_cnn_fp32_CONFIG = CONFIG=fp32

# Generate list of .mpy files
MODULE_MPYS = $(addprefix $(MODULES_PATH)/,$(addsuffix .mpy,$(MODULES)))

# Build dynamic native module
# defaults to
$(MODULES_PATH)/%.mpy:
	make -C $(or $($(*)_SRC),src/$*) \
		ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} \
		V=1 $($(*)_CONFIG) clean dist

check_unix_natmod: $(MODULE_MPYS)
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_all.py

$(PORT_DIR):
	mkdir -p $@

$(UNIX_MICROPYTHON): $(PORT_DIR)
	make -C $(MPY_DIR)/ports/unix V=1 USER_C_MODULES=$(C_MODULES_SRC_PATH) FROZEN_MANIFEST=$(MANIFEST_PATH) CFLAGS_EXTRA="-Wno-unused-function -Wno-unused-function ${CFLAGS_EXTRA}" -j4
	cp $(MPY_DIR)/ports/unix/build-standard/micropython $@

unix: $(UNIX_MICROPYTHON)

check_unix: $(UNIX_MICROPYTHON)
	$(UNIX_MICROPYTHON) tests/test_all.py test_iir,test_fft,test_arrayutils
	# TODO: enable more modules

rp2: $(PORT_DIR)
	make -C $(MPY_DIR)/ports/rp2 V=1 USER_C_MODULES=$(C_MODULES_SRC_PATH)/micropython.cmake FROZEN_MANIFEST=$(MANIFEST_PATH) CFLAGS_EXTRA='-Wno-unused-function -Wno-unused-function' -j4
	mkdir -p ./dist/ports/rp2/RPI_PICO
	cp -r $(MPY_DIR)/ports/rp2/build-RPI_PICO/firmware* ./dist/ports/rp2/RPI_PICO/


extmod:
	make -C $(MPY_DIR)/ports/esp32 V=1 BOARD=$(BOARD) USER_C_MODULES=$(C_MODULES_SRC_PATH)/micropython.cmake FROZEN_MANIFEST=$(MANIFEST_PATH) CFLAGS_EXTRA='-Wno-unused-function -Wno-unused-function' -j4
	mkdir -p $(PORT_DIST_DIR)
	cp -r $(PORT_BUILD_DIR)/firmware* $(PORT_DIST_DIR)
	cp -r $(PORT_BUILD_DIR)/micropython* $(PORT_DIST_DIR)
	

.PHONY: clean unix

clean:
	make -C src/emlearn_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C src/emlearn_neighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C src/emlearn_iir/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	rm -rf ./dist

RELEASE_NAME = emlearn-micropython-$(VERSION)
# NOTE: does not depend on dist.
release:
	mkdir $(RELEASE_NAME)
	cp -r dist/* $(RELEASE_NAME) 
	zip -r $(RELEASE_NAME).zip $(RELEASE_NAME)
	#cp $(RELEASE_NAME).zip emlearn-micropython-latest.zip

check: check_unix_natmod

dist: $(MODULE_MPYS)


