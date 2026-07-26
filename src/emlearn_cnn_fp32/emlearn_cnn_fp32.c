/*
 * emlearn_cnn_fp32 wrapper
 * This file includes fp32/tm_port.h directly before including mod_cnn.c
 * We need to modify mod_cnn.c to not include tm_port.h when CONFIG is already defined
 */

/* Define CONFIG_FP32 first */
#define CONFIG_FP32

// for external module we need static
#define TM_STATIC static

/* Include the fp32 tm_port.h directly */
#include "../tinymaix_cnn/fp32/tm_port.h"

/* Now include mod_cnn.c - it will see CONFIG_FP32 is defined and use fp32/tm_port.h */
#include "../tinymaix_cnn/mod_cnn.c"
