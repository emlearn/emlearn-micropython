
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

$(MODULES_PATH)/emlearn_trees.mpy:
	make -C src/emlearn_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_neighbors.mpy:
	make -C src/emlearn_neighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_iir.mpy:
	make -C src/emlearn_iir/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_fft.mpy:
	make -C src/emlearn_fft/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_cnn_int8.mpy:
	make -C src/tinymaix_cnn/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 CONFIG=int8 clean dist

$(MODULES_PATH)/emlearn_cnn_fp32.mpy:
	make -C src/tinymaix_cnn/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 CONFIG=fp32 clean dist

$(MODULES_PATH)/emlearn_kmeans.mpy:
	make -C src/emlearn_kmeans/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_iir_q15.mpy:
	make -C src/emlearn_iir_q15/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_arrayutils.mpy:
	make -C src/emlearn_arrayutils/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

$(MODULES_PATH)/emlearn_linreg.mpy:
	make -C src/emlearn_linreg/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) CFLAGS_EXTRA=${CFLAGS_EXTRA} V=1 clean dist

emlearn_trees.results: $(MODULES_PATH)/emlearn_trees.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_trees.py

emlearn_neighbors.results: $(MODULES_PATH)/emlearn_neighbors.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_neighbors.py

emlearn_iir.results: $(MODULES_PATH)/emlearn_iir.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_iir.py

emlearn_fft.results: $(MODULES_PATH)/emlearn_fft.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_fft.py

emlearn_cnn.results: $(MODULES_PATH)/emlearn_cnn_int8.mpy $(MODULES_PATH)/emlearn_cnn_fp32.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_cnn.py

emlearn_kmeans.results: $(MODULES_PATH)/emlearn_kmeans.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_kmeans.py

emlearn_iir_q15.results: $(MODULES_PATH)/emlearn_iir_q15.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_iir_q15.py

emlearn_arrayutils.results: $(MODULES_PATH)/emlearn_arrayutils.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_arrayutils.py

emlearn_linreg.results: $(MODULES_PATH)/emlearn_linreg.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_linreg.py

tests_all.results:
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_all.py

$(PORT_DIR):
	mkdir -p $@

$(UNIX_MICROPYTHON): $(PORT_DIR)
	make -C $(MPY_DIR)/ports/unix V=1 USER_C_MODULES=$(C_MODULES_SRC_PATH) FROZEN_MANIFEST=$(MANIFEST_PATH) CFLAGS_EXTRA='-Wno-unused-function -Wno-unused-function' -j4
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

check: tests_all.results

dist: $(MODULES_PATH)/emlearn_trees.mpy $(MODULES_PATH)/emlearn_neighbors.mpy $(MODULES_PATH)/emlearn_iir.mpy $(MODULES_PATH)/emlearn_iir_q15.mpy $(MODULES_PATH)/emlearn_fft.mpy $(MODULES_PATH)/emlearn_kmeans.mpy $(MODULES_PATH)/emlearn_arrayutils.mpy $(MODULES_PATH)/emlearn_cnn_int8.mpy $(MODULES_PATH)/emlearn_cnn_fp32.mpy $(MODULES_PATH)/emlearn_linreg.mpy


