// Include the header file to get access to the MicroPython API
#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <string.h>

// NOTE: make sure we do not use sqrtf() wrapper which uses errno, does not work in native module
#if USE_IEEE_SQRTF
#define sqrtf(x) __ieee754_sqrtf(x)
#elif USE_BUILTIN_SQRTF
#define sqrtf(x) __builtin_sqrtf(x)
#endif

#include "eml_sefr.c"

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

static float *sefr_get_float_buffer(mp_obj_t obj, mp_buffer_info_t *info, mp_uint_t flags) {
    mp_get_buffer_raise(obj, info, flags);
    if (info->typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    return (float *)info->buf;
}

// MicroPython type for the SEFR model
typedef struct _mp_obj_sefr_model_t {
    mp_obj_base_t base;
    sefr_model_t model;
    bool fitted;
    bool lda_started;
} mp_obj_sefr_model_t;

#if MICROPY_ENABLE_DYNRUNTIME
mp_obj_full_type_t sefr_model_type;
#else
static const mp_obj_type_t sefr_model_type;
#endif

// Create a new instance
static mp_obj_t sefr_model_new(mp_obj_t n_features_obj, mp_obj_t n_classes_obj) {
    mp_int_t n_features = mp_obj_get_int(n_features_obj);
    mp_int_t n_classes = mp_obj_get_int(n_classes_obj);
    if (n_features < 1 || n_features > UINT16_MAX || n_classes < 2 || n_classes > UINT16_MAX) {
        mp_raise_ValueError(MP_ERROR_TEXT("need n_features >= 1 and n_classes >= 2"));
    }

    mp_obj_sefr_model_t *o = mp_obj_malloc(mp_obj_sefr_model_t, (mp_obj_type_t *)&sefr_model_type);
    sefr_model_t *self = &o->model;
    self->n_features = n_features;
    self->n_classes = n_classes;

    const size_t weight_count = (size_t)n_features * (size_t)n_classes;
    self->class_sums = m_new(float, weight_count);
    self->weights = m_new(float, weight_count);
    self->feature_min = m_new(float, n_features);
    self->feature_max = m_new(float, n_features);
    self->class_counts = m_new(uint32_t, n_classes);
    self->biases = m_new(float, n_classes);
    self->lda_scatter = m_new(float, (size_t)n_classes * (size_t)n_classes);
    self->lda_means = m_new(float, (size_t)n_classes * (size_t)n_classes);
    self->lda_temp = m_new(float, 2 * (size_t)n_classes);

    sefr_reset(self);
    o->fitted = false;
    o->lda_started = false;

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_2(sefr_model_new_obj, sefr_model_new);

// Delete an instance
static mp_obj_t sefr_model_del(mp_obj_t self_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_model_t *self = &o->model;
    const size_t weight_count = (size_t)self->n_features * (size_t)self->n_classes;

    m_del(float, self->class_sums, weight_count);
    m_del(float, self->weights, weight_count);
    m_del(float, self->feature_min, self->n_features);
    m_del(float, self->feature_max, self->n_features);
    m_del(uint32_t, self->class_counts, self->n_classes);
    m_del(float, self->biases, self->n_classes);
    m_del(float, self->lda_scatter, (size_t)self->n_classes * (size_t)self->n_classes);
    m_del(float, self->lda_means, (size_t)self->n_classes * (size_t)self->n_classes);
    m_del(float, self->lda_temp, 2 * (size_t)self->n_classes);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(sefr_model_del_obj, sefr_model_del);

// Forget all data seen so far
static mp_obj_t sefr_model_reset(mp_obj_t self_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_reset(&o->model);
    o->fitted = false;
    o->lda_started = false;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(sefr_model_reset_obj, sefr_model_reset);

// Add a batch of samples. X is flattened (n_samples * n_features), y holds class indices
static mp_obj_t sefr_model_partial_fit(mp_obj_t self_obj, mp_obj_t X_obj, mp_obj_t y_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_model_t *self = &o->model;

    mp_buffer_info_t X_info;
    mp_buffer_info_t y_info;
    const float *X = sefr_get_float_buffer(X_obj, &X_info, MP_BUFFER_READ);
    const float *y = sefr_get_float_buffer(y_obj, &y_info, MP_BUFFER_READ);
    const size_t X_len = X_info.len / sizeof(float);
    const size_t n_samples = y_info.len / sizeof(float);

    if (X_len != n_samples * self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X size must be len(y) * n_features"));
    }
    if (!sefr_partial_fit(self, X, y, n_samples)) {
        mp_raise_ValueError(MP_ERROR_TEXT("labels must be integers in [0, n_classes)"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(sefr_model_partial_fit_obj, sefr_model_partial_fit);

// Compute the weights from the data seen so far. Can be called again after more partial_fit()
static mp_obj_t sefr_model_finalize(mp_obj_t self_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    if (!sefr_finalize(&o->model)) {
        mp_raise_ValueError(MP_ERROR_TEXT("need samples from at least two classes"));
    }
    o->fitted = true;
    o->lda_started = false;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(sefr_model_finalize_obj, sefr_model_finalize);

// Optional second pass over the same data, after finalize():
// linear discriminant analysis on the class scores. Call for every batch, then finalize_lda()
static mp_obj_t sefr_model_partial_fit_lda(mp_obj_t self_obj, mp_obj_t X_obj, mp_obj_t y_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_model_t *self = &o->model;
    if (!o->fitted) {
        mp_raise_ValueError(MP_ERROR_TEXT("call finalize() before partial_fit_lda()"));
    }
    if (!o->lda_started) {
        if (!sefr_lda_begin(self)) {
            mp_raise_ValueError(MP_ERROR_TEXT("LDA needs samples of every class"));
        }
        o->lda_started = true;
    }

    mp_buffer_info_t X_info;
    mp_buffer_info_t y_info;
    const float *X = sefr_get_float_buffer(X_obj, &X_info, MP_BUFFER_READ);
    const float *y = sefr_get_float_buffer(y_obj, &y_info, MP_BUFFER_READ);
    const size_t n_samples = y_info.len / sizeof(float);
    if (X_info.len / sizeof(float) != n_samples * self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X size must be len(y) * n_features"));
    }
    if (!sefr_lda_partial_fit(self, X, y, n_samples)) {
        mp_raise_ValueError(MP_ERROR_TEXT("labels must be integers in [0, n_classes)"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(sefr_model_partial_fit_lda_obj, sefr_model_partial_fit_lda);

static mp_obj_t sefr_model_finalize_lda(mp_obj_t self_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    if (!o->lda_started || !sefr_lda_finalize(&o->model)) {
        mp_raise_ValueError(MP_ERROR_TEXT("LDA failed: call partial_fit_lda() with data first"));
    }
    o->lda_started = false;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(sefr_model_finalize_lda_obj, sefr_model_finalize_lda);

// Predict one sample: fills scores (n_classes), returns the predicted class
static mp_obj_t sefr_model_predict(mp_obj_t self_obj, mp_obj_t features_obj, mp_obj_t scores_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_model_t *self = &o->model;
    if (!o->fitted) {
        mp_raise_ValueError(MP_ERROR_TEXT("call finalize() before predict()"));
    }

    mp_buffer_info_t features_info;
    mp_buffer_info_t scores_info;
    const float *features = sefr_get_float_buffer(features_obj, &features_info, MP_BUFFER_READ);
    float *scores = sefr_get_float_buffer(scores_obj, &scores_info, MP_BUFFER_WRITE);

    if (features_info.len / sizeof(float) != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }
    if (scores_info.len / sizeof(float) != self->n_classes) {
        mp_raise_ValueError(MP_ERROR_TEXT("scores buffer must have n_classes entries"));
    }

    return mp_obj_new_int(sefr_predict(self, features, scores));
}
static MP_DEFINE_CONST_FUN_OBJ_3(sefr_model_predict_obj, sefr_model_predict);

static mp_obj_t sefr_model_get_weights(mp_obj_t self_obj, mp_obj_t out_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_model_t *self = &o->model;

    mp_buffer_info_t info;
    float *out = sefr_get_float_buffer(out_obj, &info, MP_BUFFER_WRITE);
    const size_t length = (size_t)self->n_features * (size_t)self->n_classes;
    if (info.len / sizeof(float) != length) {
        mp_raise_ValueError(MP_ERROR_TEXT("Buffer is wrong size"));
    }
    memcpy(out, self->weights, sizeof(float) * length);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(sefr_model_get_weights_obj, sefr_model_get_weights);

static mp_obj_t sefr_model_get_bias(mp_obj_t self_obj, mp_obj_t out_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    sefr_model_t *self = &o->model;

    mp_buffer_info_t info;
    float *out = sefr_get_float_buffer(out_obj, &info, MP_BUFFER_WRITE);
    if (info.len / sizeof(float) != self->n_classes) {
        mp_raise_ValueError(MP_ERROR_TEXT("Buffer is wrong size"));
    }
    memcpy(out, self->biases, sizeof(float) * self->n_classes);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(sefr_model_get_bias_obj, sefr_model_get_bias);

static mp_obj_t sefr_model_get_n_features(mp_obj_t self_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    return mp_obj_new_int(o->model.n_features);
}
static MP_DEFINE_CONST_FUN_OBJ_1(sefr_model_get_n_features_obj, sefr_model_get_n_features);

static mp_obj_t sefr_model_get_n_classes(mp_obj_t self_obj) {
    mp_obj_sefr_model_t *o = MP_OBJ_TO_PTR(self_obj);
    return mp_obj_new_int(o->model.n_classes);
}
static MP_DEFINE_CONST_FUN_OBJ_1(sefr_model_get_n_classes_obj, sefr_model_get_n_classes);

#if MICROPY_ENABLE_DYNRUNTIME

mp_map_elem_t sefr_model_locals_dict_table[11];
static MP_DEFINE_CONST_DICT(sefr_model_locals_dict, sefr_model_locals_dict_table);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&sefr_model_new_obj));

    sefr_model_type.base.type = (void *)&mp_fun_table.type_type;
    sefr_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    sefr_model_type.name = MP_QSTR_sefr;

    sefr_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_partial_fit), MP_OBJ_FROM_PTR(&sefr_model_partial_fit_obj) };
    sefr_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_finalize), MP_OBJ_FROM_PTR(&sefr_model_finalize_obj) };
    sefr_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&sefr_model_predict_obj) };
    sefr_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_reset), MP_OBJ_FROM_PTR(&sefr_model_reset_obj) };
    sefr_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&sefr_model_del_obj) };
    sefr_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_weights), MP_OBJ_FROM_PTR(&sefr_model_get_weights_obj) };
    sefr_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_bias), MP_OBJ_FROM_PTR(&sefr_model_get_bias_obj) };
    sefr_model_locals_dict_table[7] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_features), MP_OBJ_FROM_PTR(&sefr_model_get_n_features_obj) };
    sefr_model_locals_dict_table[8] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_classes), MP_OBJ_FROM_PTR(&sefr_model_get_n_classes_obj) };

    sefr_model_locals_dict_table[9] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_partial_fit_lda), MP_OBJ_FROM_PTR(&sefr_model_partial_fit_lda_obj) };
    sefr_model_locals_dict_table[10] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_finalize_lda), MP_OBJ_FROM_PTR(&sefr_model_finalize_lda_obj) };

    MP_OBJ_TYPE_SET_SLOT(&sefr_model_type, locals_dict, (void *)&sefr_model_locals_dict, 10);

    MP_DYNRUNTIME_INIT_EXIT
}

#else

static const mp_rom_map_elem_t sefr_model_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_partial_fit), MP_ROM_PTR(&sefr_model_partial_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_finalize), MP_ROM_PTR(&sefr_model_finalize_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&sefr_model_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR_reset), MP_ROM_PTR(&sefr_model_reset_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&sefr_model_del_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_weights), MP_ROM_PTR(&sefr_model_get_weights_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_bias), MP_ROM_PTR(&sefr_model_get_bias_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_features), MP_ROM_PTR(&sefr_model_get_n_features_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_classes), MP_ROM_PTR(&sefr_model_get_n_classes_obj) },
    { MP_ROM_QSTR(MP_QSTR_partial_fit_lda), MP_ROM_PTR(&sefr_model_partial_fit_lda_obj) },
    { MP_ROM_QSTR(MP_QSTR_finalize_lda), MP_ROM_PTR(&sefr_model_finalize_lda_obj) },
};
static MP_DEFINE_CONST_DICT(sefr_model_locals_dict, sefr_model_locals_dict_table);

static MP_DEFINE_CONST_OBJ_TYPE(
    sefr_model_type,
    MP_QSTR_sefr,
    MP_TYPE_FLAG_ITER_IS_CUSTOM,
    locals_dict, &sefr_model_locals_dict
);

static const mp_rom_map_elem_t sefr_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_new), MP_ROM_PTR(&sefr_model_new_obj) },
    { MP_ROM_QSTR(MP_QSTR_sefr), MP_ROM_PTR(&sefr_model_type) },
};
static MP_DEFINE_CONST_DICT(sefr_globals, sefr_globals_table);

const mp_obj_module_t sefr_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&sefr_globals,
};
MP_REGISTER_MODULE(MP_QSTR_emlearn_sefr_c, sefr_cmodule);

#endif
