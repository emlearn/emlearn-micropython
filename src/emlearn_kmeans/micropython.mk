MOD_DIR := $(USERMOD_DIR)

# Add all C files to SRC_USERMOD.
SRC_USERMOD_C += $(MOD_DIR)/kmeans.c

# We can add our module folder to include paths if needed
CFLAGS_USERMOD += -I$(MOD_DIR)
