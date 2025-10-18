MOD_DIR := $(USERMOD_DIR)

# Add all C files to SRC_USERMOD.
SRC_USERMOD_C += $(MOD_DIR)/neighbors.c

EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

# We can add our module folder to include paths if needed
CFLAGS_USERMOD += -I$(EMLEARN_DIR)
