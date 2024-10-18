
ARCH ?= x64
MPY_ABI_VERSION ?= 6.3
MPY_DIR ?= ../micropython
MICROPYTHON_BIN ?= micropython

VERSION := $(shell git describe --tags --always)

MPY_DIR_ABS = $(abspath $(MPY_DIR)) 

MODULES_PATH = ./dist/$(ARCH)_$(MPY_ABI_VERSION)

$(MODULES_PATH)/emlearn_trees.mpy:
	make -C src/emlearn_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlearn_neighbors.mpy:
	make -C src/emlearn_neighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emliir.mpy:
	make -C src/emliir/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlearn_fft.mpy:
	make -C src/emlearn_fft/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlearn_cnn.mpy:
	make -C src/tinymaix_cnn/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlearn_kmeans.mpy:
	make -C src/emlearn_kmeans/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/eml_iir_q15.mpy:
	make -C src/eml_iir_q15/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlearn_arrayutils.mpy:
	make -C src/emlearn_arrayutils/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

emlearn_trees.results: $(MODULES_PATH)/emlearn_trees.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_trees.py

emlearn_neighbors.results: $(MODULES_PATH)/emlearn_neighbors.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_neighbors.py

emliir.results: $(MODULES_PATH)/emliir.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_iir.py

emlearn_fft.results: $(MODULES_PATH)/emlearn_fft.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_fft.py

emlearn_cnn.results: $(MODULES_PATH)/emlearn_cnn.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_cnn.py

emlearn_kmeans.results: $(MODULES_PATH)/emlearn_kmeans.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_kmeans.py

eml_iir_q15.results: $(MODULES_PATH)/eml_iir_q15.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_iir_q15.py

emlearn_arrayutils.results: $(MODULES_PATH)/emlearn_arrayutils.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_arrayutils.py

.PHONY: clean

clean:
	make -C src/emlearn_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C src/emlearn_neighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C src/emliir/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	rm -rf ./dist

RELEASE_NAME = emlearn-micropython-$(VERSION)
# NOTE: does not depend on dist.
release:
	mkdir $(RELEASE_NAME)
	cp -r dist/* $(RELEASE_NAME) 
	zip -r $(RELEASE_NAME).zip $(RELEASE_NAME)
	#cp $(RELEASE_NAME).zip emlearn-micropython-latest.zip

check: emlearn_trees.results emlearn_neighbors.results emliir.results eml_iir_q15.results emlearn_fft.results emlearn_kmeans.results emlearn_arrayutils.results emlearn_cnn.results

dist: $(MODULES_PATH)/emlearn_trees.mpy $(MODULES_PATH)/emlearn_neighbors.mpy $(MODULES_PATH)/emliir.mpy $(MODULES_PATH)/eml_iir_q15.mpy $(MODULES_PATH)/emlearn_fft.mpy $(MODULES_PATH)/emlearn_kmeans.mpy $(MODULES_PATH)/emlearn_arrayutils.mpy $(MODULES_PATH)/emlearn_cnn.mpy


