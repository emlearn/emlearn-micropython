// MicroPython module wrapper for PLS Regression
// Supports both native module (dynruntime) and external C module builds

#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <string.h>
#include <stdbool.h>

// NOTE: make sure we do not use sqrtf() wrapper which uses errno, does not work in native module
#if USE_IEEE_SQRTF
#define sqrtf(x) __ieee754_sqrtf(x)
#elif USE_BUILTIN_SQRTF
#define sqrtf(x) __builtin_sqrtf(x)
#endif

#include "eml_plsr.h"

#ifdef MICROPY_ENABLE_DYNRUNTIME
// memset/memcpy for compatibility
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif
#endif

// MicroPython type for PLSR model
typedef struct _mp_obj_plsr_model_t {
    mp_obj_base_t base;
    eml_plsr_t model;
    uint8_t *memory;  // Allocated memory block
    size_t memory_size;
    uint16_t n_samples;
    uint16_t n_features;
    uint16_t n_components;
} mp_obj_plsr_model_t;

#if MICROPY_ENABLE_DYNRUNTIME
mp_obj_full_type_t plsr_model_type;
#else
static const mp_obj_type_t plsr_model_type;
#endif

// Create a new instance
static mp_obj_t plsr_model_new(size_t n_args, const mp_obj_t *args) {
    // Args: n_samples, n_features, n_components
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: n_samples, n_features, n_components"));
    }

    mp_int_t n_samples = mp_obj_get_int(args[0]);
    mp_int_t n_features = mp_obj_get_int(args[1]);
    mp_int_t n_components = mp_obj_get_int(args[2]);

    // Validate dimensions
    if (n_samples <= 0 || n_features <= 0 || n_components <= 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Dimensions must be positive"));
    }
    if (n_components > n_features || n_components > n_samples) {
        mp_raise_ValueError(MP_ERROR_TEXT("n_components must be <= min(n_samples, n_features)"));
    }

    // Allocate space
    mp_obj_plsr_model_t *o =
        mp_obj_malloc(mp_obj_plsr_model_t, (mp_obj_type_t *)&plsr_model_type);

    o->n_samples = n_samples;
    o->n_features = n_features;
    o->n_components = n_components;

    // Calculate and allocate memory
    size_t memory_size = eml_plsr_get_memory_size(n_samples, n_features, n_components);
    o->memory_size = memory_size;
    o->memory = m_new(uint8_t, memory_size);

    if (!o->memory) {
        mp_raise_ValueError(MP_ERROR_TEXT("Failed to allocate PLSR memory"));
    }

    // Initialize model
    EmlError err = eml_plsr_init(&o->model, n_samples, n_features, n_components,
                                  o->memory, memory_size);

    if (err != EmlOk) {
        m_del(uint8_t, o->memory, memory_size);
        mp_raise_ValueError(MP_ERROR_TEXT("Failed to initialize PLSR model"));
    }

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(plsr_model_new_obj, 3, 3, plsr_model_new);

// Delete an instance
static mp_obj_t plsr_model_del(mp_obj_t self_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);

    // Free allocated memory
    if (o->memory) {
        m_del(uint8_t, o->memory, o->memory_size);
        o->memory = NULL;
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(plsr_model_del_obj, plsr_model_del);

// Start iterative fitting
static mp_obj_t plsr_model_fit_start(size_t n_args, const mp_obj_t *args) {
    // Args: self, X, y
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, X, y"));
    }

    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(args[0]);
    eml_plsr_t *self = &o->model;

    // Extract X buffer
    mp_buffer_info_t X_bufinfo;
    mp_get_buffer_raise(args[1], &X_bufinfo, MP_BUFFER_READ);
    if (X_bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("X expecting float32 array"));
    }
    const float *X = X_bufinfo.buf;
    const int X_len = X_bufinfo.len / sizeof(float);

    // Extract y buffer
    mp_buffer_info_t y_bufinfo;
    mp_get_buffer_raise(args[2], &y_bufinfo, MP_BUFFER_READ);
    if (y_bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("y expecting float32 array"));
    }
    const float *y = y_bufinfo.buf;
    const int y_len = y_bufinfo.len / sizeof(float);

    // Validate dimensions
    if (X_len != o->n_samples * o->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X dimensions don't match model"));
    }
    if (y_len != o->n_samples) {
        mp_raise_ValueError(MP_ERROR_TEXT("y dimensions don't match model"));
    }

    // Start training
    EmlError err = eml_plsr_fit_start(self, X, y);

    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("Failed to start PLSR training"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(plsr_model_fit_start_obj, 3, 3, plsr_model_fit_start);

// Single iteration step
static mp_obj_t plsr_model_step(size_t n_args, const mp_obj_t *args) {
    // Args: self, tolerance (optional)
    if (n_args < 1 || n_args > 2) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 1-2 arguments: self, [tolerance]"));
    }

    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(args[0]);
    eml_plsr_t *self = &o->model;

    // Get tolerance
    float tolerance = 1e-6f;  // Default
    if (n_args >= 2) {
        tolerance = mp_obj_get_float_to_f(args[1]);
    }

    // Perform iteration step
    EmlError err = eml_plsr_iteration_step(self, tolerance);

    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("PLSR iteration failed"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(plsr_model_step_obj, 1, 2, plsr_model_step);

// Finalize component
static mp_obj_t plsr_model_finalize_component(mp_obj_t self_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    eml_plsr_t *self = &o->model;

    EmlError err = eml_plsr_finalize_component(self);

    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("Failed to finalize component"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(plsr_model_finalize_component_obj, plsr_model_finalize_component);

// Check if converged
static mp_obj_t plsr_model_is_converged(mp_obj_t self_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    eml_plsr_t *self = &o->model;

    bool converged = eml_plsr_is_converged(self);

    return mp_obj_new_bool(converged);
}
static MP_DEFINE_CONST_FUN_OBJ_1(plsr_model_is_converged_obj, plsr_model_is_converged);

// Check if complete
static mp_obj_t plsr_model_is_complete(mp_obj_t self_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    eml_plsr_t *self = &o->model;

    bool complete = eml_plsr_is_complete(self);

    return mp_obj_new_bool(complete);
}
static MP_DEFINE_CONST_FUN_OBJ_1(plsr_model_is_complete_obj, plsr_model_is_complete);

// Predict using the model
static mp_obj_t plsr_model_predict(size_t n_args, const mp_obj_t *args) {
    // Args: self, features
    if (n_args != 2) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 2 arguments: self, features"));
    }

    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(args[0]);
    eml_plsr_t *self = &o->model;

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_READ);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    const float *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(float);

    if (n_features != o->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    // Make prediction
    float prediction;
    EmlError err = eml_plsr_predict(self, features, &prediction);

    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("Prediction failed"));
    }

    return mp_obj_new_float_from_f(prediction);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(plsr_model_predict_obj, 2, 2, plsr_model_predict);

// Get convergence metric
static mp_obj_t plsr_model_get_convergence_metric(mp_obj_t self_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    eml_plsr_t *self = &o->model;

    return mp_obj_new_float_from_f(self->convergence_metric);
}
static MP_DEFINE_CONST_FUN_OBJ_1(plsr_model_get_convergence_metric_obj, plsr_model_get_convergence_metric);

// Set auto-centering
static mp_obj_t plsr_model_set_auto_center(mp_obj_t self_obj, mp_obj_t value_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    o->model.auto_center = mp_obj_is_true(value_obj);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(plsr_model_set_auto_center_obj, plsr_model_set_auto_center);

// Get auto-centering
static mp_obj_t plsr_model_get_auto_center(mp_obj_t self_obj) {
    mp_obj_plsr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    return mp_obj_new_bool(o->model.auto_center);
}
static MP_DEFINE_CONST_FUN_OBJ_1(plsr_model_get_auto_center_obj, plsr_model_get_auto_center);

// ============================================================================
// Build-type specific module registration
// ============================================================================

#if MICROPY_ENABLE_DYNRUNTIME

// Forward declaration for locals dict
mp_map_elem_t plsr_model_locals_dict_table[10];
static MP_DEFINE_CONST_DICT(plsr_model_locals_dict, plsr_model_locals_dict_table);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&plsr_model_new_obj));

    plsr_model_type.base.type = (void*)&mp_fun_table.type_type;
    plsr_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    plsr_model_type.name = MP_QSTR_plsr;

    // methods
    plsr_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&plsr_model_predict_obj) };
    plsr_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&plsr_model_del_obj) };
    plsr_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_fit_start), MP_OBJ_FROM_PTR(&plsr_model_fit_start_obj) };
    plsr_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_step), MP_OBJ_FROM_PTR(&plsr_model_step_obj) };
    plsr_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_finalize_component), MP_OBJ_FROM_PTR(&plsr_model_finalize_component_obj) };
    plsr_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_is_converged), MP_OBJ_FROM_PTR(&plsr_model_is_converged_obj) };
    plsr_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_is_complete), MP_OBJ_FROM_PTR(&plsr_model_is_complete_obj) };
    plsr_model_locals_dict_table[7] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_convergence_metric), MP_OBJ_FROM_PTR(&plsr_model_get_convergence_metric_obj) };
    plsr_model_locals_dict_table[8] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_set_auto_center), MP_OBJ_FROM_PTR(&plsr_model_set_auto_center_obj) };
    plsr_model_locals_dict_table[9] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_auto_center), MP_OBJ_FROM_PTR(&plsr_model_get_auto_center_obj) };

    MP_OBJ_TYPE_SET_SLOT(&plsr_model_type, locals_dict, (void*)&plsr_model_locals_dict, 10);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

#else

// External C module build

static const mp_rom_map_elem_t plsr_model_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&plsr_model_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&plsr_model_del_obj) },
    { MP_ROM_QSTR(MP_QSTR_fit_start), MP_ROM_PTR(&plsr_model_fit_start_obj) },
    { MP_ROM_QSTR(MP_QSTR_step), MP_ROM_PTR(&plsr_model_step_obj) },
    { MP_ROM_QSTR(MP_QSTR_finalize_component), MP_ROM_PTR(&plsr_model_finalize_component_obj) },
    { MP_ROM_QSTR(MP_QSTR_is_converged), MP_ROM_PTR(&plsr_model_is_converged_obj) },
    { MP_ROM_QSTR(MP_QSTR_is_complete), MP_ROM_PTR(&plsr_model_is_complete_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_convergence_metric), MP_ROM_PTR(&plsr_model_get_convergence_metric_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_auto_center), MP_ROM_PTR(&plsr_model_set_auto_center_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_auto_center), MP_ROM_PTR(&plsr_model_get_auto_center_obj) },
};
static MP_DEFINE_CONST_DICT(plsr_model_locals_dict, plsr_model_locals_dict_table);

static MP_DEFINE_CONST_OBJ_TYPE(
    plsr_model_type,
    MP_QSTR_plsr,
    MP_TYPE_FLAG_ITER_IS_CUSTOM,
    make_new, plsr_model_new,
    locals_dict, &plsr_model_locals_dict
);

static const mp_rom_map_elem_t plsr_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_new), MP_ROM_PTR(&plsr_model_new_obj) },
    { MP_ROM_QSTR(MP_QSTR_plsr), MP_ROM_PTR(&plsr_model_type) },
};
static MP_DEFINE_CONST_DICT(plsr_globals, plsr_globals_table);

const mp_obj_module_t plsr_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&plsr_globals,
};
MP_REGISTER_MODULE(MP_QSTR_emlearn_plsr_c, plsr_cmodule);

#endif
