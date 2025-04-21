// Include the header file to get access to the MicroPython API
#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <eml_iir.h>

#include <string.h>

#ifdef MICROPY_ENABLE_DYNRUNTIME
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
#endif
#endif


// MicroPython type for EmlIIR
typedef struct _mp_obj_iir_filter_t {
    mp_obj_base_t base;
    EmlIIR filter;
} mp_obj_iir_filter_t;

#if MICROPY_ENABLE_DYNRUNTIME
mp_obj_full_type_t iir_filter_type;
#else
static const mp_obj_type_t iir_filter_type;
#endif

// Create a new instance
static mp_obj_t iir_filter_new(mp_obj_t array_obj) {

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(array_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
    }
    float *values = bufinfo.buf;
    const int n_values = bufinfo.len / sizeof(*values);

    if ((n_values % 6) != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Filter coefficients must be multiple of 6"));
    }

    // Construct object
    mp_obj_iir_filter_t *o = mp_obj_malloc(mp_obj_iir_filter_t, (mp_obj_type_t *)&iir_filter_type);
    EmlIIR *self = &o->filter;
    memset(self, 0, sizeof(EmlIIR)); // HACK: try to get memset symbol in

    self->n_stages = n_values / 6;

    self->states_length = self->n_stages * 4;
    self->states = m_new(float, self->states_length);

    self->coefficients_length = n_values;
    self->coefficients = m_new(float, self->coefficients_length);
    memcpy((float *)self->coefficients, values, sizeof(float)*self->coefficients_length);


    const EmlError err = eml_iir_check(*self);
    if (err != EmlOk) {
        m_del(float, self->states, self->states_length);
        m_del(float, (float *)self->coefficients, self->coefficients_length);
        mp_raise_ValueError(MP_ERROR_TEXT("EmlError"));
    }

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_1(iir_filter_new_obj, iir_filter_new);

// Delete the instance
static mp_obj_t iir_filter_del(mp_obj_t self_obj) {

    mp_obj_iir_filter_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlIIR *self = &o->filter;

    // free allocated data
    m_del(float, self->states, self->states_length);
    m_del(float, (float *)self->coefficients, self->coefficients_length);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(iir_filter_del_obj, iir_filter_del);


// Add a node to the tree
static mp_obj_t iir_filter_run(mp_obj_t self_obj, mp_obj_t array_obj) {

    mp_obj_iir_filter_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlIIR *self = &o->filter;    

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(array_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
    }
    float *values = bufinfo.buf;
    const int n_values = bufinfo.len / sizeof(*values);

    for (int i=0; i<n_values; i++) {
        float out = eml_iir_filter(*self, values[i]);
        values[i] = out;
    }

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_2(iir_filter_run_obj, iir_filter_run);


#ifdef MICROPY_ENABLE_DYNRUNTIME
mp_map_elem_t iir_locals_dict_table[2];
static MP_DEFINE_CONST_DICT(iir_locals_dict, iir_locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&iir_filter_new_obj));

    iir_filter_type.base.type = (void*)&mp_fun_table.type_type;
    iir_filter_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    iir_filter_type.name = MP_QSTR_emliir;
    // methods
    iir_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_run), MP_OBJ_FROM_PTR(&iir_filter_run_obj) };
    iir_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&iir_filter_del_obj) };

    MP_OBJ_TYPE_SET_SLOT(&iir_filter_type, locals_dict, (void*)&iir_locals_dict, 2);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}
#else


// Define a class
static const mp_rom_map_elem_t emlearn_iir_filter_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_run), MP_ROM_PTR(&iir_filter_run_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&iir_filter_del_obj) }
};
static MP_DEFINE_CONST_DICT(emlearn_iir_filter_locals_dict, emlearn_iir_filter_locals_dict_table);

static MP_DEFINE_CONST_OBJ_TYPE(
    iir_filter_type,
    MP_QSTR_emliir,
    MP_TYPE_FLAG_NONE,
    locals_dict, &emlearn_iir_filter_locals_dict
);

// Define module object.
static const mp_rom_map_elem_t emlearn_iir_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_new), MP_ROM_PTR(&iir_filter_new_obj) },
};
static MP_DEFINE_CONST_DICT(emlearn_iir_globals, emlearn_iir_globals_table);

const mp_obj_module_t emlearn_iir_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&emlearn_iir_globals,
};

MP_REGISTER_MODULE(MP_QSTR_emlearn_iir, emlearn_iir_cmodule);


#endif
