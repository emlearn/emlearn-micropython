# emlearn_cnn_fp32 wrapper for Unix port
# This wrapper sets CONFIG_FP32 before including mod_cnn.c

CNN_SRC := $(USERMOD_DIR)/../tinymaix_cnn

# Add wrapper C file which defines CONFIG_FP32
SRC_USERMOD_C += $(USERMOD_DIR)/emlearn_cnn_fp32.c

# Include paths - config directory first to ensure correct tm_port.h is found
CFLAGS_USERMOD += -I$(CNN_SRC)/fp32
CFLAGS_USERMOD += -I$(CNN_SRC)/int8
CFLAGS_USERMOD += -I$(CNN_SRC)/../../dependencies/TinyMaix/include
CFLAGS_USERMOD += -I$(CNN_SRC)/../../dependencies/TinyMaix/src

# Compile flags to suppress TinyMaix warnings
CFLAGS_USERMOD += -Wno-error=unused-variable -Wno-error=multichar -Wdouble-promotion
