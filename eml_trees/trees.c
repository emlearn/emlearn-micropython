// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include "model.h"
//#include <eml_trees.h>


// This is the function which will be called from Python, as factorial(x)
STATIC mp_obj_t factorial_func(mp_obj_t x_obj) {
    // Extract the integer from the MicroPython input object
    mp_int_t x = mp_obj_get_int(x_obj);

    const int result = x + 1;

    // Convert the result to a MicroPython integer object and return it
    return mp_obj_new_int(result);
}
// Define a Python reference to the function above
STATIC MP_DEFINE_CONST_FUN_OBJ_1(factorial_obj, factorial_func);

// FIXME: function for creating a model instance
// FIXME: function for adding decision nodes

// Takes a float array
STATIC mp_obj_t predict(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // Check number of arguments is valid
    mp_arg_check_num(n_args, n_kw, 1, 1, false);

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
    }

    float *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(*features);

    // FIXME: call model predict
    const int result = simple_rgb_pink_yellow_other_predict(features, n_features);

    return mp_obj_new_int(result);
}


// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    // Make the function available in the module's namespace
    mp_store_global(MP_QSTR_factorial, MP_OBJ_FROM_PTR(&factorial_obj));

    mp_store_global(MP_QSTR_predict, MP_DYNRUNTIME_MAKE_FUNCTION(predict));

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

