// Include the header file to get access to the MicroPython API
#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <string.h>
#include <float.h>
#include <stdbool.h>

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

static float *logreg_get_float_buffer(mp_obj_t obj,
                                      mp_buffer_info_t *info,
                                      mp_uint_t flags) {
    mp_get_buffer_raise(obj, info, flags);
    if (info->typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("workspace buffers must be float32 arrays"));
    }
    if ((info->len % sizeof(float)) != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("workspace buffers must be float32 arrays"));
    }
    return (float *)info->buf;
}

static void logreg_validate_length(uint16_t expected,
                                   uint16_t actual,
                                   const char *name) {
    if (expected != actual) {
        mp_raise_ValueError(MP_ERROR_TEXT("workspace buffer wrong length"));
    }
}

static void logreg_softmax_inplace(float *buffer, uint16_t n_classes) {
    if (n_classes == 0 || buffer == NULL) {
        return;
    }

    float max_logit = buffer[0];
    for (uint16_t cls = 1; cls < n_classes; cls++) {
        if (buffer[cls] > max_logit) {
            max_logit = buffer[cls];
        }
    }

    float sum = 0.0f;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        buffer[cls] = expf_compat(buffer[cls] - max_logit);
        sum += buffer[cls];
    }

    if (sum == 0.0f) {
        return;
    }

    const float inv_sum = 1.0f / sum;
    for (uint16_t cls = 0; cls < n_classes; cls++) {
        buffer[cls] *= inv_sum;
    }
}

// MicroPython type for logistic regression model
typedef struct _mp_obj_logreg_model_t {
    mp_obj_base_t base;
    logreg_model_t model;
} mp_obj_logreg_model_t;

#if MICROPY_ENABLE_DYNRUNTIME
mp_obj_full_type_t logreg_model_type;
#else
static const mp_obj_type_t logreg_model_type;
#endif

// Create a new instance
static mp_obj_t logreg_model_new(size_t n_args, const mp_obj_t *args) {
    // Args: n_features, n_classes, learning_rate, lambda_l2, lambda_l1
    if (n_args != 5) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 5 arguments: n_features, n_classes, learning_rate, lambda_l2, lambda_l1"));
    }

    mp_int_t n_features = mp_obj_get_int(args[0]);
    mp_int_t n_classes = mp_obj_get_int(args[1]);
    float learning_rate = mp_obj_get_float_to_f(args[2]);
    float lambda_l2 = mp_obj_get_float_to_f(args[3]);
    float lambda_l1 = mp_obj_get_float_to_f(args[4]);

    mp_obj_logreg_model_t *o =
        mp_obj_malloc(mp_obj_logreg_model_t, (mp_obj_type_t *)&logreg_model_type);

    logreg_model_t *self = &o->model;
    memset(self, 0, sizeof(logreg_model_t));

    self->n_features = n_features;
    self->n_classes = n_classes;
    self->learning_rate = learning_rate;
    self->lambda_l2 = lambda_l2;
    self->lambda_l1 = lambda_l1;

    const size_t weight_count = (size_t)n_features * (size_t)n_classes;
    self->weights = m_new(float, weight_count);
    self->weight_gradients = m_new(float, weight_count);
    self->biases = m_new(float, n_classes);

    memset(self->weights, 0, weight_count * sizeof(float));
    memset(self->biases, 0, n_classes * sizeof(float));

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(logreg_model_new_obj, 5, 5, logreg_model_new);

// Delete an instance
static mp_obj_t logreg_model_del(mp_obj_t self_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    m_del(float, self->weights, (size_t)self->n_features * (size_t)self->n_classes);
    m_del(float, self->weight_gradients, (size_t)self->n_features * (size_t)self->n_classes);
    m_del(float, self->biases, self->n_classes);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(logreg_model_del_obj, logreg_model_del);

static void logreg_parse_feature_buffer(mp_obj_t obj,
                                        mp_buffer_info_t *info,
                                        const char *name) {
    mp_get_buffer_raise(obj, info, MP_BUFFER_READ);
    if (info->typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("feature/target buffers must be float32 arrays"));
    }
    if ((info->len % sizeof(float)) != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("feature/target buffers must be float32 arrays"));
    }
}

// Single training iteration
static mp_obj_t logreg_model_step(size_t n_args, const mp_obj_t *args) {
    if (n_args != 6) {
        mp_raise_ValueError(MP_ERROR_TEXT("expected 6 arguments: self, X, y, logits, probabilities, bias_gradients"));
    }

    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(args[0]);
    logreg_model_t *self = &o->model;

    mp_obj_t X_obj = args[1];
    mp_obj_t y_obj = args[2];
    mp_obj_t logits_obj = args[3];
    mp_obj_t probs_obj = args[4];
    mp_obj_t bias_obj = args[5];

    mp_buffer_info_t X_bufinfo;
    logreg_parse_feature_buffer(X_obj, &X_bufinfo, "X");
    const float *X = X_bufinfo.buf;
    const size_t X_len = X_bufinfo.len / sizeof(float);

    mp_buffer_info_t y_bufinfo;
    logreg_parse_feature_buffer(y_obj, &y_bufinfo, "y");
    const float *y = y_bufinfo.buf;
    const size_t y_len = y_bufinfo.len / sizeof(float);

    const size_t sample_stride = (size_t)self->n_features;
    const size_t target_stride = (size_t)self->n_classes;

    if (sample_stride == 0 || target_stride == 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Model not initialised"));
    }

    if (X_len % sample_stride != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("X size mismatch"));
    }
    const size_t n_samples = X_len / sample_stride;
    if (y_len != n_samples * target_stride) {
        mp_raise_ValueError(MP_ERROR_TEXT("y size mismatch"));
    }

    mp_buffer_info_t logits_info;
    mp_buffer_info_t probs_info;
    mp_buffer_info_t bias_info;
    float *logits = logreg_get_float_buffer(logits_obj, &logits_info, MP_BUFFER_RW);
    float *probabilities = logreg_get_float_buffer(probs_obj, &probs_info, MP_BUFFER_RW);
    float *bias_gradients = logreg_get_float_buffer(bias_obj, &bias_info, MP_BUFFER_RW);

    const uint16_t logits_len = logits_info.len / sizeof(float);
    const uint16_t probs_len = probs_info.len / sizeof(float);
    const uint16_t bias_len = bias_info.len / sizeof(float);

    logreg_validate_length(self->n_classes, logits_len, "logits");
    logreg_validate_length(self->n_classes, probs_len, "probabilities");
    logreg_validate_length(self->n_classes, bias_len, "bias_gradients");

    logreg_workspace_t workspace = {
        .logits = logits,
        .probabilities = probabilities,
        .bias_gradients = bias_gradients,
        .logits_size = logits_len,
        .probabilities_size = probs_len,
        .bias_gradients_size = bias_len,
    };

    bool ok = logreg_iterate(self, X, y, (uint16_t)n_samples, &workspace);
    if (!ok) {
        mp_raise_ValueError(MP_ERROR_TEXT("iteration failed"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(logreg_model_step_obj, 6, 6, logreg_model_step);

static mp_obj_t logreg_model_predict(size_t n_args, const mp_obj_t *args) {
    if (n_args != 4) {
        mp_raise_ValueError(MP_ERROR_TEXT("expected 4 arguments: self, features, probabilities, logits"));
    }

    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(args[0]);
    logreg_model_t *self = &o->model;

    mp_obj_t features_obj = args[1];
    mp_obj_t probs_obj = args[2];
    mp_obj_t logits_obj = args[3];

    mp_buffer_info_t feat_buf;
    logreg_parse_feature_buffer(features_obj, &feat_buf, "features");
    const float *features = feat_buf.buf;
    const int n_features = feat_buf.len / sizeof(float);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    mp_buffer_info_t probs_info;
    mp_buffer_info_t logits_info;
    float *probs = logreg_get_float_buffer(probs_obj, &probs_info, MP_BUFFER_RW);
    float *logits = logreg_get_float_buffer(logits_obj, &logits_info, MP_BUFFER_RW);

    const uint16_t probs_len = probs_info.len / sizeof(float);
    const uint16_t logits_len = logits_info.len / sizeof(float);

    logreg_validate_length(self->n_classes, probs_len, "probabilities");
    logreg_validate_length(self->n_classes, logits_len, "logits");

    logreg_predict_scores(self, features, logits);
    memcpy(probs, logits, sizeof(float) * self->n_classes);
    logreg_softmax_inplace(probs, self->n_classes);

    uint16_t best_cls = 0;
    float best_value = probs[0];
    for (uint16_t cls = 1; cls < self->n_classes; cls++) {
        if (probs[cls] > best_value) {
            best_value = probs[cls];
            best_cls = cls;
        }
    }

    return mp_obj_new_int(best_cls);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(logreg_model_predict_obj, 4, 4, logreg_model_predict);

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
    const int length = bufinfo.len / sizeof(float);

    if (length != self->n_features * self->n_classes) {
        mp_raise_ValueError(MP_ERROR_TEXT("Buffer is wrong size"));
    }

    memcpy(weights, self->weights, sizeof(float) * length);

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
    const int length = bufinfo.len / sizeof(float);

    if (length != self->n_features * self->n_classes) {
        mp_raise_ValueError(MP_ERROR_TEXT("Weight array size mismatch"));
    }

    memcpy(self->weights, weights, sizeof(float) * length);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_set_weights_obj, logreg_model_set_weights);

static mp_obj_t logreg_model_get_bias(mp_obj_t self_obj, mp_obj_t out_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(out_obj, &bufinfo, MP_BUFFER_WRITE);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    const size_t n_out = bufinfo.len / sizeof(float);
    if (n_out != self->n_classes) {
        mp_raise_ValueError(MP_ERROR_TEXT("bias buffer wrong length"));
    }

    float *out = bufinfo.buf;
    memcpy(out, self->biases, sizeof(float) * self->n_classes);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_get_bias_obj, logreg_model_get_bias);

static mp_obj_t logreg_model_set_bias(mp_obj_t self_obj, mp_obj_t bias_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t bufinfo;
    if (mp_get_buffer(bias_obj, &bufinfo, MP_BUFFER_READ)) {
        if (bufinfo.typecode != 'f') {
            mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
        }
        const size_t n_bias = bufinfo.len / sizeof(float);
        if (n_bias != self->n_classes) {
            mp_raise_ValueError(MP_ERROR_TEXT("bias array size mismatch"));
        }
        const float *biases = bufinfo.buf;
        memcpy(self->biases, biases, sizeof(float) * self->n_classes);
    } else {
        size_t len;
        mp_obj_t *items;
        mp_obj_get_array(bias_obj, &len, &items);
        if (len != self->n_classes) {
            mp_raise_ValueError(MP_ERROR_TEXT("bias array size mismatch"));
        }
        for (size_t i = 0; i < len; i++) {
            self->biases[i] = mp_obj_get_float_to_f(items[i]);
        }
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(logreg_model_set_bias_obj, logreg_model_set_bias);

static mp_obj_t logreg_model_get_n_features(mp_obj_t self_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    return mp_obj_new_int(self->n_features);
}
static MP_DEFINE_CONST_FUN_OBJ_1(logreg_model_get_n_features_obj, logreg_model_get_n_features);

static mp_obj_t logreg_model_get_n_classes(mp_obj_t self_obj) {
    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(self_obj);
    logreg_model_t *self = &o->model;

    return mp_obj_new_int(self->n_classes);
}
static MP_DEFINE_CONST_FUN_OBJ_1(logreg_model_get_n_classes_obj, logreg_model_get_n_classes);

static mp_obj_t logreg_model_score_logloss(size_t n_args, const mp_obj_t *args) {
    if (n_args != 5) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 5 arguments: self, X, y, logits, probabilities"));
    }

    mp_obj_logreg_model_t *o = MP_OBJ_TO_PTR(args[0]);
    logreg_model_t *self = &o->model;

    mp_buffer_info_t X_bufinfo;
    logreg_parse_feature_buffer(args[1], &X_bufinfo, "X");
    const float *X = X_bufinfo.buf;
    const size_t X_len = X_bufinfo.len / sizeof(float);

    mp_buffer_info_t y_bufinfo;
    logreg_parse_feature_buffer(args[2], &y_bufinfo, "y");
    const float *y = y_bufinfo.buf;
    const size_t y_len = y_bufinfo.len / sizeof(float);

    const size_t sample_stride = (size_t)self->n_features;
    const size_t target_stride = (size_t)self->n_classes;
    if (sample_stride == 0 || target_stride == 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Model not initialised"));
    }
    if (X_len % sample_stride != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("X size mismatch"));
    }
    const size_t n_samples = X_len / sample_stride;
    if (y_len != n_samples * target_stride) {
        mp_raise_ValueError(MP_ERROR_TEXT("y size mismatch"));
    }

    mp_buffer_info_t logits_info;
    mp_buffer_info_t probs_info;
    float *logits = logreg_get_float_buffer(args[3], &logits_info, MP_BUFFER_RW);
    float *probabilities = logreg_get_float_buffer(args[4], &probs_info, MP_BUFFER_RW);

    const uint16_t logits_len = logits_info.len / sizeof(float);
    const uint16_t probs_len = probs_info.len / sizeof(float);

    logreg_validate_length(self->n_classes, logits_len, "logits");
    logreg_validate_length(self->n_classes, probs_len, "probabilities");

    logreg_workspace_t workspace = {
        .logits = logits,
        .probabilities = probabilities,
        .bias_gradients = NULL,
        .logits_size = logits_len,
        .probabilities_size = probs_len,
        .bias_gradients_size = 0,
    };

    float loss_value = 0.0f;
    bool ok = logreg_logloss(self, X, y, (uint16_t)n_samples, &workspace, &loss_value);

    if (!ok) {
        mp_raise_ValueError(MP_ERROR_TEXT("logloss failed"));
    }

    return mp_obj_new_float_from_f(loss_value);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(logreg_model_score_logloss_obj, 5, 5, logreg_model_score_logloss);

#if MICROPY_ENABLE_DYNRUNTIME

// Forward declaration for locals dict
mp_map_elem_t logreg_model_locals_dict_table[10];
static MP_DEFINE_CONST_DICT(logreg_model_locals_dict, logreg_model_locals_dict_table);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&logreg_model_new_obj));

    logreg_model_type.base.type = (void *)&mp_fun_table.type_type;
    logreg_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    logreg_model_type.name = MP_QSTR_logreg;

    logreg_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&logreg_model_predict_obj) };
    logreg_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_step), MP_OBJ_FROM_PTR(&logreg_model_step_obj) };
    logreg_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&logreg_model_del_obj) };
    logreg_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_weights), MP_OBJ_FROM_PTR(&logreg_model_get_weights_obj) };
    logreg_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_set_weights), MP_OBJ_FROM_PTR(&logreg_model_set_weights_obj) };
    logreg_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_bias), MP_OBJ_FROM_PTR(&logreg_model_get_bias_obj) };
    logreg_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_set_bias), MP_OBJ_FROM_PTR(&logreg_model_set_bias_obj) };
    logreg_model_locals_dict_table[7] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_features), MP_OBJ_FROM_PTR(&logreg_model_get_n_features_obj) };
    logreg_model_locals_dict_table[8] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_classes), MP_OBJ_FROM_PTR(&logreg_model_get_n_classes_obj) };
    logreg_model_locals_dict_table[9] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_score_logloss), MP_OBJ_FROM_PTR(&logreg_model_score_logloss_obj) };

    MP_OBJ_TYPE_SET_SLOT(&logreg_model_type, locals_dict, (void *)&logreg_model_locals_dict, 10);

    MP_DYNRUNTIME_INIT_EXIT
}

#else

static const mp_rom_map_elem_t logreg_model_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&logreg_model_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR_step), MP_ROM_PTR(&logreg_model_step_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&logreg_model_del_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_weights), MP_ROM_PTR(&logreg_model_get_weights_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_weights), MP_ROM_PTR(&logreg_model_set_weights_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_bias), MP_ROM_PTR(&logreg_model_get_bias_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_bias), MP_ROM_PTR(&logreg_model_set_bias_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_features), MP_ROM_PTR(&logreg_model_get_n_features_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_classes), MP_ROM_PTR(&logreg_model_get_n_classes_obj) },
    { MP_ROM_QSTR(MP_QSTR_score_logloss), MP_ROM_PTR(&logreg_model_score_logloss_obj) },
};
static MP_DEFINE_CONST_DICT(logreg_model_locals_dict, logreg_model_locals_dict_table);

static MP_DEFINE_CONST_OBJ_TYPE(
    logreg_model_type,
    MP_QSTR_logreg,
    MP_TYPE_FLAG_ITER_IS_CUSTOM,
    make_new, logreg_model_new,
    locals_dict, &logreg_model_locals_dict
);

static const mp_rom_map_elem_t logreg_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_new), MP_ROM_PTR(&logreg_model_new_obj) },
    { MP_ROM_QSTR(MP_QSTR_logreg), MP_ROM_PTR(&logreg_model_type) },
};
static MP_DEFINE_CONST_DICT(logreg_globals, logreg_globals_table);

const mp_obj_module_t logreg_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&logreg_globals,
};
MP_REGISTER_MODULE(MP_QSTR_emlearn_logreg_c, logreg_cmodule);

#endif
