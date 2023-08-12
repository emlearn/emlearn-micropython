
ARCH := x64
MPY_ABI_VERSION := 6.1
MPY_DIR := ../micropython
MICROPYTHON := $(MPY_DIR)/ports/unix/build-standard/micropython

VERSION := $(shell git describe --tags --always)

MPY_DIR_ABS = $(abspath $(MPY_DIR)) 

MODULES_PATH = ./dist/$(ARCH)_$(MPY_ABI_VERSION)

$(MODULES_PATH)/emltrees.mpy:
	make -C eml_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

$(MODULES_PATH)/emlneighbors.mpy:
	make -C emlneighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean dist

emltrees.results: $(MODULES_PATH)/emltrees.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON) tests/test_trees.py

emlneighbors.results: $(MODULES_PATH)/emlneighbors.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON) tests/test_neighbors.py

.PHONY: clean

clean:
	make -C eml_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	make -C emlneighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR_ABS) V=1 clean
	rm -rf ./dist

RELEASE_NAME = emlearn-micropython-$(VERSION)
# NOTE: does not depend on dist.
release:
	mkdir $(RELEASE_NAME)
	cp -r dist/* $(RELEASE_NAME) 
	zip $(RELEASE_NAME).zip $(RELEASE_NAME)
	#cp $(RELEASE_NAME).zip emlearn-micropython-latest.zip

check: emltrees.results emlneighbors.results

dist: $(MODULES_PATH)/emltrees.mpy $(MODULES_PATH)/emlneighbors.mpy


