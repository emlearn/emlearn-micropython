// Include the header file to get access to the MicroPython API
#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <string.h>

#include "eml_logreg.c"

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

// MicroPython type for logistic regression model
typedef struct _mp_obj_logreg_model_t {
    mp_obj_base_t base;
    logreg_model_t model;
} mp_obj_logreg_model_t;

#ifdef MICROPY_ENABLE_DYNRUNTIME
mp_obj_full_type_t logreg_model_type;
#else
static const mp_obj_type_t logreg_model_type;
#endif

// Forward declaration for locals dict
mp_map_elem_t logreg_model_locals_dict_table[10];
static MP_DEFINE_CONST_DICT(logreg_model_locals_dict, logreg_model_locals_dict_table);

// Create a new instance
static mp_obj_t logreg_model_new(size_t n_args, const mp_obj_t *args) {
    // Args: n_features, learning_rate, lambda_l2, lambda_l1
    if (n_args != 4) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 4 arguments: n_features, learning_rate, lambda_l2, lambda_l1"));
    }

    mp_int_t n_features = mp_obj_get_int(args[0]);
    float learning_rate = mp_obj_get_float_to_f(args[1]);
    float lambda_l2 = mp_obj_get_float_to_f(args[2]);
    float lambda_l1 = mp_obj_get_float_to_f(args[3]);

    mp_obj_logreg_model_t *o =
        mp_obj_malloc(mp_obj_logreg_model_t, (mp_obj_type_t *)&logreg_model_type);

    logreg_model_t *self = &o->model;
    memset(self, 0, sizeof(logreg_model_t));

    self->n_features = n_features;
    self->learning_rate = learning_rate;
    self->lambda_l2 = lambda_l2;
    self->lambda_l1 = lambda_l1;
    self->bias = 0.0f;

    self->weights = (float *)m_malloc(sizeof(float) * n_features);
    self->weight_gradients = (float *)m_malloc(sizeof(float) * n_features);

    memset(self->weights, 0, n_features * sizeof(float));

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(logreg_model_new_obj, 4, 4, logreg_model_new);

// Delete an instance
static mp_obj_t logreg_model_del(mp_obj_t self_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    m_free(self->weights);
    m_free(self->weight_gradients);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(logreg_model_del_obj, logreg_model_del);

// Single training iteration
static mp_obj_t logreg_model_step(mp_obj_t self_obj, mp_obj_t X_obj, mp_obj_t y_obj) {

    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t X_bufinfo;
    mp_get_buffer_raise(X_obj, &X_bufinfo, MP_BUFFER_READ);
    if (X_bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("X expecting float32 array"));
    }
    const float *X = X_bufinfo.buf;
    const int X_len = X_bufinfo.len / sizeof(float);

    mp_buffer_info_t y_bufinfo;
    mp_get_buffer_raise(y_obj, &y_bufinfo, MP_BUFFER_READ);
    if (y_bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("y expecting float32 array"));
    }
    const float *y = y_bufinfo.buf;
    const int y_len = y_bufinfo.len / sizeof(float);

    if (X_len != y_len * self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X and y dimensions don't match"));
    }

    const uint16_t n_samples = y_len;
    logreg_iterate(self, X, y, n_samples);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(logreg_model_step_obj, logreg_model_step);

static mp_obj_t logreg_model_predict(mp_obj_t self_obj, mp_obj_t features_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(features_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    const float *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(float);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    float prediction = logreg_predict_proba(self, features);

    return mp_obj_new_float_from_f(prediction);
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_predict_obj, logreg_model_predict);

static mp_obj_t logreg_model_predict_class(mp_obj_t self_obj, mp_obj_t features_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(features_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    const float *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(float);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    const float threshold = 0.5f;

    uint8_t label = logreg_predict_proba(self, features) >= threshold ? 1 : 0;

    return mp_obj_new_int(label);
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_predict_class_obj, logreg_model_predict_class);

// Get model weights
static mp_obj_t logreg_model_get_weights(mp_obj_t self_obj, mp_obj_t out_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(out_obj, &bufinfo, MP_BUFFER_WRITE);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    float *weights = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(float);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Buffer is wrong size"));
    }

    memcpy(weights, self->weights, sizeof(float) * n_features);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_get_weights_obj, logreg_model_get_weights);

static mp_obj_t logreg_model_set_weights(mp_obj_t self_obj, mp_obj_t weights_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(weights_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    const float *weights = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(float);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Weight array size mismatch"));
    }

    memcpy(self->weights, weights, sizeof(float) * n_features);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_set_weights_obj, logreg_model_set_weights);

static mp_obj_t logreg_model_get_bias(mp_obj_t self_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    return mp_obj_new_float_from_f(self->bias);
}
static MP_DEFINE_CONST_FUN_OBJ_1(logreg_model_get_bias_obj, logreg_model_get_bias);

static mp_obj_t logreg_model_set_bias(mp_obj_t self_obj, mp_obj_t bias_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    float bias = mp_obj_get_float_to_f(bias_obj);
    self->bias = bias;

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_set_bias_obj, logreg_model_set_bias);

static mp_obj_t logreg_model_get_n_features(mp_obj_t self_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    return mp_obj_new_int(self->n_features);
}
static MP_DEFINE_CONST_FUN_OBJ_1(logreg_model_get_n_features_obj, logreg_model_get_n_features);

static mp_obj_t logreg_model_score_logloss(size_t n_args, const mp_obj_t *args) {
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, X, y"));
    }

    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(args[0]);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t X_bufinfo;
    mp_get_buffer_raise(args[1], &X_bufinfo, MP_BUFFER_READ);
    if (X_bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("X expecting float32 array"));
    }
    const float *X = X_bufinfo.buf;
    const int X_len = X_bufinfo.len / sizeof(float);

    mp_buffer_info_t y_bufinfo;
    mp_get_buffer_raise(args[2], &y_bufinfo, MP_BUFFER_READ);
    if (y_bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("y expecting float32 array"));
    }
    const float *y = y_bufinfo.buf;
    const int y_len = y_bufinfo.len / sizeof(float);

    if (X_len != y_len * self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X and y dimensions don't match"));
    }

    const uint16_t n_samples = y_len;
    float loss = logreg_logloss(self, X, y, n_samples);

    return mp_obj_new_float_from_f(loss);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(logreg_model_score_logloss_obj, 3, 3, logreg_model_score_logloss);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&logreg_model_new_obj));

    logreg_model_type.base.type = (void *)&mp_fun_table.type_type;
    logreg_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    logreg_model_type.name = MP_QSTR_logreg;

    logreg_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&logreg_model_predict_obj) };
    logreg_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict_class), MP_OBJ_FROM_PTR(&logreg_model_predict_class_obj) };
    logreg_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_step), MP_OBJ_FROM_PTR(&logreg_model_step_obj) };
    logreg_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&logreg_model_del_obj) };
    logreg_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_weights), MP_OBJ_FROM_PTR(&logreg_model_get_weights_obj) };
    logreg_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_set_weights), MP_OBJ_FROM_PTR(&logreg_model_set_weights_obj) };
    logreg_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_bias), MP_OBJ_FROM_PTR(&logreg_model_get_bias_obj) };
    logreg_model_locals_dict_table[7] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_set_bias), MP_OBJ_FROM_PTR(&logreg_model_set_bias_obj) };
    logreg_model_locals_dict_table[8] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_features), MP_OBJ_FROM_PTR(&logreg_model_get_n_features_obj) };
    logreg_model_locals_dict_table[9] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_score_logloss), MP_OBJ_FROM_PTR(&logreg_model_score_logloss_obj) };

    MP_OBJ_TYPE_SET_SLOT(&logreg_model_type, locals_dict, (void *)&logreg_model_locals_dict, 10);

    MP_DYNRUNTIME_INIT_EXIT
}
