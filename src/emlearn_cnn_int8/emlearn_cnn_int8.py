# emlearn_cnn_int8 - wrapper for frozen unix build
# Imports the native C module compiled with int8 configuration
try:
    from emlearn_cnn_int8_native import *
except ImportError:
    pass
