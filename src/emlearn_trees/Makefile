
# Location of emlearn library
EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

# Name of module
MOD = emlearn_trees

# Source files (.c or .py)
SRC = trees.c emlearn_trees.py

# Include to get the rules for compiling and linking the module
include ../dynmodule.mk

# Must come after dynmodule include
CFLAGS += -I$(EMLEARN_DIR) -Wno-unused-function
