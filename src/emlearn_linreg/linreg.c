// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include <string.h>

#include "eml_linreg.c"

// memset/memcpy for compatibility 
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif

// MicroPython type for ElasticNet model
typedef struct _mp_obj_elasticnet_model_t {
    mp_obj_base_t base;
    elastic_net_model_t model;
} mp_obj_elasticnet_model_t;

mp_obj_full_type_t elasticnet_model_type;

// Create a new instance
static mp_obj_t elasticnet_model_new(size_t n_args, const mp_obj_t *args) {
    // Args: n_features, alpha, l1_ratio, learning_rate
    if (n_args != 4) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 4 arguments: n_features, alpha, l1_ratio, learning_rate"));
    }
    
    mp_int_t n_features = mp_obj_get_int(args[0]);
    float alpha = mp_obj_get_float(args[1]);
    float l1_ratio = mp_obj_get_float(args[2]);
    float learning_rate = mp_obj_get_float(args[3]);

    // Allocate space
    mp_obj_elasticnet_model_t *o = \
        mp_obj_malloc(mp_obj_elasticnet_model_t, (mp_obj_type_t *)&elasticnet_model_type);

    elastic_net_model_t *self = &o->model;
    memset(self, 0, sizeof(elastic_net_model_t));

    // Configure model
    self->n_features = n_features;
    self->alpha = alpha;
    self->l1_ratio = l1_ratio;
    self->learning_rate = learning_rate;
    self->bias = 0.0f;
    
    // Allocate weight buffers
    self->weights = (float *)m_malloc(sizeof(float) * n_features);
    self->weight_gradients = (float *)m_malloc(sizeof(float) * n_features);
    
    // Initialize weights to zero
    memset(self->weights, 0, n_features * sizeof(float));

    return MP_OBJ_FROM_PTR(o);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(elasticnet_model_new_obj, 4, 4, elasticnet_model_new);

// Delete an instance
static mp_obj_t elasticnet_model_del(mp_obj_t self_obj) {
    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(self_obj);
    elastic_net_model_t *self = &o->model;   

    // Free allocated memory
    m_free(self->weights);
    m_free(self->weight_gradients);

    return mp_const_none;
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(elasticnet_model_del_obj, elasticnet_model_del);

// Single training iteration
static mp_obj_t elasticnet_model_step(size_t n_args, const mp_obj_t *args) {
    // Args: self, X, y
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, X, y"));
    }
    
    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(args[0]);
    elastic_net_model_t *self = &o->model;

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
    if (X_len != y_len * self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X and y dimensions don't match"));
    }

    const uint16_t n_samples = y_len;

    // Perform single iteration
    elastic_net_iterate(self, X, y, n_samples);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(elasticnet_model_step_obj, 3, 3, elasticnet_model_step);

// Predict using the model
static mp_obj_t elasticnet_model_predict(mp_obj_fun_bc_t *self_obj,
        size_t n_args, size_t n_kw, mp_obj_t *args) {
    // Check number of arguments is valid
    mp_arg_check_num(n_args, n_kw, 2, 2, false);

    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(args[0]);
    elastic_net_model_t *self = &o->model;    

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_READ);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float32 array"));
    }
    const float *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(float);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    // Make prediction
    float prediction = predict_sample(self, features);

    return mp_obj_new_float(prediction);
}

// Get model weights
static mp_obj_t elasticnet_model_get_weights(mp_obj_t self_obj, mp_obj_t out_obj) {
    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(self_obj);
    elastic_net_model_t *self = &o->model;

    // Extract buffer pointer and verify typecode
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
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_2(elasticnet_model_get_weights_obj, elasticnet_model_get_weights);

// Get number of features
static mp_obj_t elasticnet_model_get_n_features(mp_obj_t self_obj) {
    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(self_obj);
    elastic_net_model_t *self = &o->model;

    return mp_obj_new_int(self->n_features);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(elasticnet_model_get_n_features_obj, elasticnet_model_get_n_features);

// Get model bias
static mp_obj_t elasticnet_model_get_bias(mp_obj_t self_obj) {
    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(self_obj);
    elastic_net_model_t *self = &o->model;

    return mp_obj_new_float(self->bias);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(elasticnet_model_get_bias_obj, elasticnet_model_get_bias);

// Calculate MSE from X and y (saves memory by not storing predictions)
static mp_obj_t elasticnet_model_score_mse(size_t n_args, const mp_obj_t *args) {
    // Args: self, X, y
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, X, y"));
    }
    
    mp_obj_elasticnet_model_t *o = MP_OBJ_TO_PTR(args[0]);
    elastic_net_model_t *self = &o->model;

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
    if (X_len != y_len * self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X and y dimensions don't match"));
    }

    const uint16_t n_samples = y_len;
    float mse = elastic_net_mse(self, X, y, n_samples);

    return mp_obj_new_float(mse);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(elasticnet_model_score_mse_obj, 3, 3, elasticnet_model_score_mse);

// Module setup
mp_map_elem_t elasticnet_model_locals_dict_table[8];
static MP_DEFINE_CONST_DICT(elasticnet_model_locals_dict, elasticnet_model_locals_dict_table);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&elasticnet_model_new_obj));

    elasticnet_model_type.base.type = (void*)&mp_fun_table.type_type;
    elasticnet_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    elasticnet_model_type.name = MP_QSTR_elasticnet;
    
    // methods
    elasticnet_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_DYNRUNTIME_MAKE_FUNCTION(elasticnet_model_predict) };
    elasticnet_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_step), MP_OBJ_FROM_PTR(&elasticnet_model_step_obj) };
    elasticnet_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&elasticnet_model_del_obj) };
    elasticnet_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_weights), MP_OBJ_FROM_PTR(&elasticnet_model_get_weights_obj) };
    elasticnet_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_bias), MP_OBJ_FROM_PTR(&elasticnet_model_get_bias_obj) };
    elasticnet_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_features), MP_OBJ_FROM_PTR(&elasticnet_model_get_n_features_obj) };
    elasticnet_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_score_mse), MP_OBJ_FROM_PTR(&elasticnet_model_score_mse_obj) };

    MP_OBJ_TYPE_SET_SLOT(&elasticnet_model_type, locals_dict, (void*)&elasticnet_model_locals_dict, 7);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}
