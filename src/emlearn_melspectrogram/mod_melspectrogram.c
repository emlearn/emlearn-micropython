// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

//#include <eml_audio.h>

#include <string.h>

// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}

void NORETURN abort() {
    while (1) {
        ;
    }
}

int
__aeabi_idiv0(int return_value) {
  return return_value;
}

long long
__aeabi_ldiv0(long long return_value) {
  return return_value;
}

#endif



// MicroPython type for Processor
typedef struct _processor_type_t {
    mp_obj_base_t base;
} processor_type_t;

mp_obj_full_type_t processor_type;

// Create a new instance
static mp_obj_t processor_new(mp_obj_t array_obj) {

    // Construct object
    processor_type_t *o = mp_obj_malloc(processor_type_t, (mp_obj_type_t *)&processor_type);


    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_1(processor_new_obj, processor_new);

// Delete the instance
static mp_obj_t processor_del(mp_obj_t self_obj) {

    //processor_type_t *o = MP_OBJ_TO_PTR(self_obj);

    // free allocated data

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(processor_del_obj, processor_del);


// Add a node to the tree
static mp_obj_t processor_run(mp_obj_t self_obj, mp_obj_t samples_obj, mp_obj_t mels_obj)
{

    // processor_type_t *o = MP_OBJ_TO_PTR(self_obj);

#if 0
    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(array_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
    }
    float *values = bufinfo.buf;
    const int n_values = bufinfo.len / sizeof(*values);

    EmlAudioMel params = 

    EmlError eml_audio_melspec(EmlAudioMel mel, EmlVector spec, EmlVector mels)

    EmlError err = \
        eml_audio_melspectrogram( mel_params, EmlFFT fft, EmlVector inout, EmlVector temp);
#endif

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_3(processor_run_obj, processor_run);


mp_map_elem_t locals_dict_table[2];
static MP_DEFINE_CONST_DICT(locals_dict, locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&processor_new_obj));

    processor_type.base.type = (void*)&mp_fun_table.type_type;
    processor_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    processor_type.name = MP_QSTR_processor;
    // methods
    locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_run), MP_OBJ_FROM_PTR(&processor_run_obj) };
    locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&processor_del_obj) };

    MP_OBJ_TYPE_SET_SLOT(&processor_type, locals_dict, (void*)&locals_dict, 2);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

