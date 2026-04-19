# emlearn_cnn_fp32 wrapper for Unix port
# Compiles mod_cnn.c with fp32 configuration

CNN_SRC := $(USERMOD_DIR)/../tinymaix_cnn

# Add C source file from tinymaix_cnn
SRC_USERMOD_C += $(CNN_SRC)/mod_cnn.c

# Include paths
CFLAGS_USERMOD += -I$(CNN_SRC)/fp32
CFLAGS_USERMOD += -I$(CNN_SRC)/../../dependencies/TinyMaix/include
CFLAGS_USERMOD += -I$(CNN_SRC)/../../dependencies/TinyMaix/src

# Compile flags
CFLAGS_USERMOD += -Wno-error=unused-variable -Wno-error=multichar -Wdouble-promotion

# Define CONFIG_FP32 for the C code
CFLAGS_USERMOD += -DCONFIG_FP32
