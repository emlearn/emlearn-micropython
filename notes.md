
## Floating point in MicroPython native modules

#### Missig _get/_new_float

On some architectures without floating point enabled, such as `armv6m` and `armv7m`. 
```
main.c:26:29: error: implicit declaration of function 'mp_obj_get_float'; did you mean 'mp_obj_is_float'?
```

Can be solved by using `mp_obj_get_float_to_f` instead


#### Missing symbols for soft-float operations

Works on `armv7emsp` (ARM Cortex M4F/M7)

ESP32
```
make -C eml_trees/ ARCH=xtensawin

nm eml_trees/build/trees.o -u
         U __divsf3
```
This is part of `libgcc.a`.
https://gcc.gnu.org/onlinedocs/gccint/Soft-float-library-routines.html

https://github.com/gcc-mirror/gcc/blob/master/libgcc/soft-fp/divsf3.c

Division operation came from  `out[0] = sum / forest->n_trees;`
in `eml_trees_regress`

arm32_aeabi_softfloat.c


Cortex M0/M3
```
make -C eml_trees/ ARCH=armv6m

nm eml_trees/build/trees.o -u
         U __aeabi_f2iz
         U __aeabi_fadd
         U __aeabi_fcmplt
         U __aeabi_fdiv
         U __aeabi_i2f
```
