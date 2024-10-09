
ARCH ?= x64
MPY_ABI_VERSION ?= 6.3
MPY_DIR ?= ../micropython
MICROPYTHON_BIN ?= micropython

VERSION := $(shell git describe --tags --always)

MPY_DIR_ABS = $(abspath $(MPY_DIR)) 

MODULES_PATH = ./dist/$(ARCH)_$(MPY_ABI_VERSION)

$(MODULES_PATH)/emltrees.mpy:
	make -C src/emltrees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlneighbors.mpy:
	make -C src/emlneighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emliir.mpy:
	make -C src/emliir/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlfft.mpy:
	make -C src/emlfft/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/tinymaix_cnn.mpy:
	make -C src/tinymaix_cnn/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlkmeans.mpy:
	make -C src/emlkmeans/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/eml_iir_q15.mpy:
	make -C src/eml_iir_q15/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlearn_arrayutils.mpy:
	make -C src/emlearn_arrayutils/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

emltrees.results: $(MODULES_PATH)/emltrees.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_trees.py

emlneighbors.results: $(MODULES_PATH)/emlneighbors.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_neighbors.py

emliir.results: $(MODULES_PATH)/emliir.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_iir.py

emlfft.results: $(MODULES_PATH)/emlfft.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_fft.py

tinymaix_cnn.results: $(MODULES_PATH)/tinymaix_cnn.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_cnn.py

emlkmeans.results: $(MODULES_PATH)/emlkmeans.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_kmeans.py

eml_iir_q15.results: $(MODULES_PATH)/eml_iir_q15.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_iir_q15.py

emlearn_arrayutils.results: $(MODULES_PATH)/emlearn_arrayutils.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON_BIN) tests/test_arrayutils.py

.PHONY: clean

clean:
	make -C src/emltrees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C src/emlneighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C src/emliir/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	rm -rf ./dist

RELEASE_NAME = emlearn-micropython-$(VERSION)
# NOTE: does not depend on dist.
release:
	mkdir $(RELEASE_NAME)
	cp -r dist/* $(RELEASE_NAME) 
	zip -r $(RELEASE_NAME).zip $(RELEASE_NAME)
	#cp $(RELEASE_NAME).zip emlearn-micropython-latest.zip

check: emltrees.results emlneighbors.results emliir.results eml_iir_q15.results emlfft.results emlkmeans.results emlearn_arrayutils.results tinymaix_cnn.results

dist: $(MODULES_PATH)/emltrees.mpy $(MODULES_PATH)/emlneighbors.mpy $(MODULES_PATH)/emliir.mpy $(MODULES_PATH)/eml_iir_q15.mpy $(MODULES_PATH)/emlfft.mpy $(MODULES_PATH)/emlkmeans.mpy $(MODULES_PATH)/emlearn_arrayutils.mpy $(MODULES_PATH)/tinymaix_cnn.mpy


