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

#define TM_ARCH_CPU         (0) //default, pure cpu compute
#define TM_ARCH_ARM_SIMD    (1) //ARM Cortex M4/M7, etc.
#define TM_ARCH_ARM_NEON    (2) //ARM Cortex A7, etc.
#define TM_ARCH_ARM_MVEI    (3) //ARMv8.1: M55, etc.
#define TM_ARCH_RV32P       (4) //T-head E907, etc.
#define TM_ARCH_RV64V       (5) //T-head C906,C910, etc.
#define TM_ARCH_CSKYV2      (6) //cskyv2 with dsp core
#define TM_ARCH_X86_SSE2    (7) //x86 sse2

#define TM_OPT0             (0) //default, least code and buf
#define TM_OPT1             (1) //opt for speed, need more code and buf
#define TM_OPT2             (2) //TODO

/******************************* PORT CONFIG  ************************************/
#define TM_ARCH         TM_ARCH_CPU
#define TM_OPT_LEVEL    TM_OPT0
#define TM_MDL_TYPE     TM_MDL_INT8
#define TM_FASTSCALE    (0)         //enable if your chip don't have FPU, may speed up 1/3, but decrease accuracy
#define TM_LOCAL_MATH   (1)         //use local math func (like exp()) to avoid libm
#define TM_ENABLE_STAT  (1)         //enable mdl stat functions
#define TM_MAX_CSIZE    (1000)      //max channel num //used if INT8 mdl  //cost TM_MAX_CSIZE*4 Byte
#define TM_MAX_KSIZE    (5*5)       //max kernel_size   //cost TM_MAX_KSIZE*4 Byte
#define TM_MAX_KCSIZE   (3*3*256)   //max kernel_size*channels //cost TM_MAX_KSIZE*sizeof(mtype_t) Byte

#define TM_INLINE       __attribute__((always_inline)) static inline
#define TM_WEAK         __attribute__((weak))

// Disable "static" (non-const) globals, since they are not supported by MicroPython mpy_ld.py
#define TM_STATIC       

// Use MicroPython for dynamic allocation
#define tm_malloc(x)   m_malloc(x)
#define tm_free(x)     mod_cnn_free(x)

// FIXME: set theese to use MicroPython primitives

#define TM_PRINTF(...) //printf(__VA_ARGS__);
#define TM_DBG(...)    TM_PRINTF("###L%d: ",__LINE__);TM_PRINTF(__VA_ARGS__);
#define TM_DBGL()      //Serial.println(__LINE__);


// FIXME: enable profiling
#define TM_GET_US()       (0)
#define TM_DBGT_INIT()     ;
#define TM_DBGT_START()    ;
#define TM_DBGT(x)         ;

/******************************* DBG PERFORMANCE CONFIG  ************************************/
//need clock tick to make accurate statistics
#define TM_EN_PERF 0

#if TM_EN_PERF
    #define  TM_GET_TICK(x)     __ASM volatile("csrr %0, mcycle" : "=r"(x)); //edit your self

    #define  TM_TICK_PERUS    (380) //sysconf(_SC_CLK_TCK)/1000000)
    #define  TM_PERF_REG(x)    uint64_t x=0;
    #define  TM_PERF_EXTREG(x) extern uint64_t x;
    #define  TM_PERF_INIT(x)   uint64_t _##x##_t0, _##x##_t1;
    #define  TM_PERF_START(x)  TM_GET_TICK(_##x##_t0);
    #define  TM_PERF_ADD(x)   {TM_GET_TICK(_##x##_t1);(x)+=(_##x##_t1-_##x##_t0);TM_GET_TICK(_##x##_t0);};
    #define  TM_PERF_PRINT(x) TM_PRINTF("PERF "#x": %ld us\r\n", (x)/TM_TICK_PERUS)
#else
    #define  TM_GET_TICK(x)
    #define  TM_TICK_PERUS
    #define  TM_PERF_REG(x)
    #define  TM_PERF_EXTREG(x)
    #define  TM_PERF_INIT(x)
    #define  TM_PERF_START(x)
    #define  TM_PERF_ADD(x)
    #define  TM_PERF_PRINT(x)
#endif


/******************************* OPS CONFIG  ************************************/




#endif
