// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include "dsp/filtering_functions.h"

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
    arm_biquad_casd_df1_inst_q15 biquad;
} mp_obj_iir_filter_t;

mp_obj_full_type_t iir_filter_type;

// Create a new instance
static mp_obj_t iir_filter_new(mp_obj_t array_obj) {

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(array_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting int16/h array"));
    }
    q15_t *values = bufinfo.buf;
    const int n_values = bufinfo.len / sizeof(*values);

    if ((n_values % 6) != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Filter coefficients must be multiple of 6"));
    }

    const int post_shift = 2; // FIXME: allow to pass as argument

    // Construct object
    mp_obj_iir_filter_t *o = mp_obj_malloc(mp_obj_iir_filter_t, (mp_obj_type_t *)&iir_filter_type);
    arm_biquad_casd_df1_inst_q15 *biquad = &o->biquad;

    const int n_stages = n_values / 6;
    const int n_states = 4 * n_stages; 

    // allocate memory
    q15_t *states = (q15_t *)m_malloc(sizeof(q15_t)*n_states);
    q15_t *coefficients = (q15_t *)m_malloc(sizeof(q15_t)*n_stages);
    // initialize coefficients and states
    memcpy((q15_t *)coefficients, values, sizeof(q15_t)*n_values);
    memset(states, 0, n_states);

    arm_biquad_cascade_df1_init_q15(biquad, n_stages, coefficients, states, post_shift);

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_1(iir_filter_new_obj, iir_filter_new);

// Delete the instance
static mp_obj_t iir_filter_del(mp_obj_t self_obj) {

    mp_obj_iir_filter_t *o = MP_OBJ_TO_PTR(self_obj);
    arm_biquad_casd_df1_inst_q15 *biquad = &o->biquad;

    // free allocated data
    m_free(biquad->pState);
    m_free((void *)biquad->pCoeffs);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(iir_filter_del_obj, iir_filter_del);


// Add a node to the tree
static mp_obj_t iir_filter_run(mp_obj_t self_obj, mp_obj_t array_obj) {

    mp_obj_iir_filter_t *o = MP_OBJ_TO_PTR(self_obj);
    arm_biquad_casd_df1_inst_q15 *biquad = &o->biquad;    

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(array_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting int16/h array"));
    }
    q15_t *values = bufinfo.buf;
    const int n_values = bufinfo.len / sizeof(*values);

    // XXX: src=dst, is in-place operation supported?
    arm_biquad_cascade_df1_q15(biquad, values, values, n_values);

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_2(iir_filter_run_obj, iir_filter_run);


mp_map_elem_t locals_dict_table[2];
static MP_DEFINE_CONST_DICT(locals_dict, locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&iir_filter_new_obj));

    iir_filter_type.base.type = (void*)&mp_fun_table.type_type;
    iir_filter_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    iir_filter_type.name = MP_QSTR_emliirq15;
    // methods
    locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_run), MP_OBJ_FROM_PTR(&iir_filter_run_obj) };
    locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&iir_filter_del_obj) };

    MP_OBJ_TYPE_SET_SLOT(&iir_filter_type, locals_dict, (void*)&locals_dict, 2);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

