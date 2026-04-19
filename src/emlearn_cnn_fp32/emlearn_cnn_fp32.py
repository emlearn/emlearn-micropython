# emlearn_cnn_fp32 - wrapper for frozen unix build
# Imports the native C module compiled with fp32 configuration
try:
    from tinymaix_cnn_fp32_native import *
except ImportError:
    pass
