// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include <eml_iir.h>

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



// MicroPython type for EmlIIR
typedef struct _mp_obj_iir_filter_t {
    mp_obj_base_t base;
    EmlIIR filter;
} mp_obj_iir_filter_t;

mp_obj_full_type_t iir_filter_type;

// Create a new instance
STATIC mp_obj_t iir_filter_new(mp_obj_t array_obj) {

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
    self->states = (float *)m_malloc(sizeof(float)*self->states_length);

    self->coefficients_length = n_values;
    self->coefficients = (float *)m_malloc(sizeof(float)*self->coefficients_length);
    memcpy((float *)self->coefficients, values, sizeof(float)*self->coefficients_length);


    const EmlError err = eml_iir_check(*self);
    if (err != EmlOk) {
        m_free(self->states);
        m_free((float *)self->coefficients);
        mp_raise_ValueError(MP_ERROR_TEXT("EmlError"));
    }

    return MP_OBJ_FROM_PTR(o);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(iir_filter_new_obj, iir_filter_new);

// Delete the instance
STATIC mp_obj_t iir_filter_del(mp_obj_t self_obj) {

    mp_obj_iir_filter_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlIIR *self = &o->filter;

    // free allocated data
    m_free(self->states);
    m_free((float *)self->coefficients);


    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(iir_filter_del_obj, iir_filter_del);


// Add a node to the tree
STATIC mp_obj_t iir_filter_run(mp_obj_t self_obj, mp_obj_t array_obj) {

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
STATIC MP_DEFINE_CONST_FUN_OBJ_2(iir_filter_run_obj, iir_filter_run);


mp_map_elem_t iir_locals_dict_table[2];
STATIC MP_DEFINE_CONST_DICT(iir_locals_dict, iir_locals_dict_table);

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

