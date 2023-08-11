
ARCH := x64
MPY_ABI_VERSION := 6.1
MICROPYTHON := ../micropython/ports/unix/build-standard/micropython
MPY_DIR := ../../micropython

MODULES_PATH = ./dist/$(ARCH)_$(MPY_ABI_VERSION)

$(MODULES_PATH)/emltrees.mpy:
	make -C eml_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR) V=1 clean dist

$(MODULES_PATH)/emlneighbors.mpy:
	make -C emlneighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR) V=1 clean dist

emltrees.results: $(MODULES_PATH)/emltrees.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON) tests/test_trees.py

emlneighbors.results: $(MODULES_PATH)/emlneighbors.mpy
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON) tests/test_neighbors.py

clean:
	make -C eml_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR) V=1 clean
	make -C emlneighbors/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR) V=1 clean
	rm -rf ./dist

check: emltrees.results emlneighbors.results

dist: $(MODULES_PATH)/emltrees.mpy $(MODULES_PATH)/emlneighbors.mpy 
