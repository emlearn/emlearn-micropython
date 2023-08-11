
ARCH := x64
MPY_ABI_VERSION := 6.1
MICROPYTHON := ../micropython/ports/unix/build-standard/micropython
MPY_DIR := ../../micropython

MODULES_PATH = ./dist/$(ARCH)_$(MPY_ABI_VERSION)
EMLTREES_MPY = $(MODULES_PATH)/emltrees.mpy

$(EMLTREES_MPY):
	make -C eml_trees/ ARCH=$(ARCH) MPY_DIR=$(MPY_DIR) V=1 clean dist

emltrees.results: $(EMLTREES_MPY) tests/out
	MICROPYPATH=$(MODULES_PATH) $(MICROPYTHON) tests/test_trees.py

check: emltrees.results
