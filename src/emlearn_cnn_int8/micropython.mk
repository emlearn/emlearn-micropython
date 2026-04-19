# emlearn_cnn_int8 wrapper for Unix port
# This wrapper sets CONFIG_INT8 before including mod_cnn.c

CNN_SRC := $(USERMOD_DIR)/../tinymaix_cnn

# Add wrapper C file which defines CONFIG_INT8
SRC_USERMOD_C += $(USERMOD_DIR)/emlearn_cnn_int8.c

# Include paths - need to include tm_port.h location
CFLAGS_USERMOD += -I$(CNN_SRC)/int8
CFLAGS_USERMOD += -I$(CNN_SRC)/../../dependencies/TinyMaix/include
CFLAGS_USERMOD += -I$(CNN_SRC)/../../dependencies/TinyMaix/src

# Compile flags to suppress TinyMaix warnings
CFLAGS_USERMOD += -Wno-error=unused-variable -Wno-error=multichar -Wdouble-promotion
