/*
 * emlearn_cnn_int8 wrapper
 * This file includes int8/tm_port.h directly before including mod_cnn.c
 * We need to modify mod_cnn.c to not include tm_port.h when CONFIG is already defined
 */

/* Define CONFIG_INT8 first */
#define CONFIG_INT8

/* Include the int8 tm_port.h directly */
#include "../tinymaix_cnn/int8/tm_port.h"

/* Now include mod_cnn.c - it will see CONFIG_INT8 is defined and use int8/tm_port.h */
#include "../tinymaix_cnn/mod_cnn.c"
