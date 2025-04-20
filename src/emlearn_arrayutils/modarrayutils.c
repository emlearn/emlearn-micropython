// Include the header file to get access to the MicroPython API
#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <string.h>

// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif



#define MAP_LINEAR(x, in_min, in_max, out_min, out_max) do { \
  const float out = ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min) \
  return max(out_min, min(out, out_max)); \
}

#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

#define min(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

float
map_linear(float x, float in_min, float in_max, float out_min, float out_max) {
  const float out = ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min);
  return max(out_min, min(out, out_max));
}


int get_typecode_size(char typecode)
{
    // TODO: support int8 also
    if (typecode == 'h') {
        return 2;
    } else if (typecode == 'f') {
        return 4;
    } else {
        // unsupported
        return 0;
    }
}

/* linear_map

Map the elements of an an array linearly between one range of values to another range.
Also accepts an output array, which may be of a different type than the input.

See map() in Arduino https://www.arduino.cc/reference/en/language/functions/math/map/
 */
static mp_obj_t
arrayutils_linear_map(size_t n_args, const mp_obj_t *args) {

    // Extract arguments
    mp_buffer_info_t in_bufinfo;
    mp_get_buffer_raise(args[0], &in_bufinfo, MP_BUFFER_RW);
    mp_buffer_info_t out_bufinfo;
    mp_get_buffer_raise(args[1], &out_bufinfo, MP_BUFFER_RW);
    const float in_min = mp_obj_get_float_to_f(args[2]);
    const float in_max = mp_obj_get_float_to_f(args[3]);
    const float out_min = mp_obj_get_float_to_f(args[4]);
    const float out_max = mp_obj_get_float_to_f(args[5]);

    // Check lengths
    const int in_length = in_bufinfo.len / get_typecode_size(in_bufinfo.typecode);
    const int out_length = in_bufinfo.len / get_typecode_size(in_bufinfo.typecode);
    if (in_length == 0 || out_length == 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("unsupported array typecode"));        
    }
    if (in_length != out_length) {
        mp_raise_ValueError(MP_ERROR_TEXT("length mismatch"));
    }

    // Actually perform the conversions
    const char in_type = in_bufinfo.typecode;
    const char out_type = out_bufinfo.typecode;
    // FIXME: support f/f and h/h
    if (in_type == 'h' && out_type == 'f') {
        const int16_t *in = in_bufinfo.buf; 
        float *out = out_bufinfo.buf;
        for (int i=0; i<in_length; i++) {
            out[i] = map_linear(in[i], in_min, in_max, out_min, out_max);
        }
    } else if (in_type == 'f' && out_type == 'h') {
        const float *in = in_bufinfo.buf; 
        int16_t *out = out_bufinfo.buf;
        for (int i=0; i<in_length; i++) {
            out[i] = (int16_t)map_linear(in[i], in_min, in_max, out_min, out_max);
        }
    } else {
        mp_raise_ValueError(MP_ERROR_TEXT("unsupported array types"));
    }

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(arrayutils_linear_map_obj, 6, 6, arrayutils_linear_map);



#ifdef MICROPY_ENABLE_DYNRUNTIME
// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_linear_map, MP_OBJ_FROM_PTR(&arrayutils_linear_map_obj));

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}
#else // extmod

// Define module object.
static const mp_rom_map_elem_t arrayutils_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_linear_map), MP_ROM_PTR(&arrayutils_linear_map_obj) }
};
static MP_DEFINE_CONST_DICT(arrayutils_globals, arrayutils_globals_table);

const mp_obj_module_t arrayutils_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&arrayutils_globals,
};

MP_REGISTER_MODULE(MP_QSTR_emlearn_arrayutils, arrayutils_cmodule);
#endif





