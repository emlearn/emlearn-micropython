#include "py/dynruntime.h"

static mp_obj_t
function(size_t n_args, const mp_obj_t *args) {

    const float in_min = mp_obj_get_float_to_f(args[2]);
    const float in_max = mp_obj_get_float_to_f(args[3]);
    const float out = in_min / in_max;

    return mp_obj_new_int(out);
 }
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(function_obj, 2, 2, function);


// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_divide, MP_OBJ_FROM_PTR(&function_obj));

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}
