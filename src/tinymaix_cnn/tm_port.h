/*
Configuration file for TinyMaix

Copyright 2022 Sipeed Technology Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef __TM_PORT_H
#define __TM_PORT_H

#define TM_ARCH_OPT0        (0) //default
#define TM_ARCH_OPT1        (1)
#define TM_ARCH_OPT2        (2)
#define TM_ARCH_ARM_SIMD    (3)
#define TM_ARCH_ARM_NEON    (4)
#define TM_ARCH_RV32P       (5)
#define TM_ARCH_RV64V       (6)

/******************************* PORT CONFIG  ************************************/
#define TM_ARCH         TM_ARCH_OPT0
#define TM_MDL_TYPE     TM_MDL_INT8 
#define TM_FASTSCALE    (1)         //enable if your chip don't have FPU, may speed up 1/3, but decrease accuracy
#define TM_ENABLE_STAT  (0)         //enable mdl stat functions
#define TM_MAX_KSIZE    (5*5)       //max kernel_size

// Use MicroPython for dynamic allocation
#define tm_malloc(x)   m_malloc(x)
#define tm_free(x)     m_free(x)

// FIXME: set theese to use MicroPython primitives

#define TM_PRINTF(...) //printf(__VA_ARGS__);
#define TM_DBG(...)    TM_PRINTF("###L%d: ",__LINE__);TM_PRINTF(__VA_ARGS__);
#define TM_DBGL()      //Serial.println(__LINE__);


// FIXME: enable profiling
#define TM_GET_US()       (0)
#define TM_DBGT_INIT()     ;
#define TM_DBGT_START()    ;
#define TM_DBGT(x)         ;

/******************************* OPS CONFIG  ************************************/


#endif
