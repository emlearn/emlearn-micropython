// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include <string.h>

#include "eml_extratrees.c"

// memset/memcpy for compatibility 
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif

// MicroPython type for ExtraTrees model
typedef struct _mp_obj_extratrees_model_t {
    mp_obj_base_t base;
    EmlTreesModel model;
    EmlTreesWorkspace workspace;
    int16_t *features_buffer;  // Buffer to store features during training
    int16_t *labels_buffer;    // Buffer to store labels during training
} mp_obj_extratrees_model_t;

mp_obj_full_type_t extratrees_model_type;

// Create a new instance
static mp_obj_t extratrees_model_new(size_t n_args, const mp_obj_t *args) {
    // Args: n_features, n_classes, [n_trees], [max_depth], [min_samples_leaf], [n_thresholds], 
    //       [subsample_ratio], [feature_subsample_ratio], [max_nodes], [max_samples], [rng_seed]
    if (n_args < 2 || n_args > 11) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 2-11 arguments: n_features, n_classes, [n_trees=10], [max_depth=10], [min_samples_leaf=1], [n_thresholds=10], [subsample_ratio=1.0], [feature_subsample_ratio=1.0], [max_nodes=1000], [max_samples=1000], [rng_seed=42]"));
    }
    
    mp_int_t n_features = mp_obj_get_int(args[0]);
    mp_int_t n_classes = mp_obj_get_int(args[1]);
    mp_int_t n_trees = (n_args > 2) ? mp_obj_get_int(args[2]) : 10;
    mp_int_t max_depth = (n_args > 3) ? mp_obj_get_int(args[3]) : 10;
    mp_int_t min_samples_leaf = (n_args > 4) ? mp_obj_get_int(args[4]) : 1;
    mp_int_t n_thresholds = (n_args > 5) ? mp_obj_get_int(args[5]) : 10;
    float subsample_ratio = (n_args > 6) ? mp_obj_get_float(args[6]) : 1.0f;
    float feature_subsample_ratio = (n_args > 7) ? mp_obj_get_float(args[7]) : 1.0f;
    mp_int_t max_nodes = (n_args > 8) ? mp_obj_get_int(args[8]) : 1000;
    mp_int_t max_samples = (n_args > 9) ? mp_obj_get_int(args[9]) : 1000;
    mp_int_t rng_seed = (n_args > 10) ? mp_obj_get_int(args[10]) : 42;

    // Allocate space
    mp_obj_extratrees_model_t *o = \
        mp_obj_malloc(mp_obj_extratrees_model_t, (mp_obj_type_t *)&extratrees_model_type);

    EmlTreesModel *model = &o->model;
    EmlTreesWorkspace *workspace = &o->workspace;
    memset(model, 0, sizeof(EmlTreesModel));
    memset(workspace, 0, sizeof(EmlTreesWorkspace));

    // Configure model
    model->n_features = n_features;
    model->n_classes = n_classes;
    model->n_trees = n_trees;
    model->max_nodes = max_nodes;
    model->n_nodes_used = 0;
    
    // Configure model config
    model->config.max_depth = max_depth;
    model->config.min_samples_leaf = min_samples_leaf;
    model->config.n_thresholds = n_thresholds;
    model->config.subsample_ratio = subsample_ratio;
    model->config.feature_subsample_ratio = feature_subsample_ratio;
    model->config.rng_seed = rng_seed;
    
    // Allocate model buffers
    model->nodes = (EmlTreesNode *)m_malloc(sizeof(EmlTreesNode) * max_nodes);
    model->tree_starts = (int16_t *)m_malloc(sizeof(int16_t) * n_trees);
    
    // Allocate workspace buffers
    workspace->sample_indices = (int16_t *)m_malloc(sizeof(int16_t) * max_samples);
    workspace->feature_indices = (int16_t *)m_malloc(sizeof(int16_t) * n_features);
    workspace->min_vals = (int16_t *)m_malloc(sizeof(int16_t) * n_features);
    workspace->max_vals = (int16_t *)m_malloc(sizeof(int16_t) * n_features);
    workspace->node_stack = (NodeState *)m_malloc(sizeof(NodeState) * 100); // Stack limit
    workspace->n_samples = 0; // Will be set during training
    workspace->rng_state = rng_seed;
    
    // Allocate training data buffers
    o->features_buffer = (int16_t *)m_malloc(sizeof(int16_t) * max_samples * n_features);
    o->labels_buffer = (int16_t *)m_malloc(sizeof(int16_t) * max_samples);
    
    // Initialize nodes and tree starts
    memset(model->nodes, 0, sizeof(EmlTreesNode) * max_nodes);
    memset(model->tree_starts, 0, sizeof(int16_t) * n_trees);

    return MP_OBJ_FROM_PTR(o);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_new_obj, 2, 11, extratrees_model_new);

// Delete an instance
static mp_obj_t extratrees_model_del(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesModel *model = &o->model;
    EmlTreesWorkspace *workspace = &o->workspace;

    // Free allocated memory
    m_free(model->nodes);
    m_free(model->tree_starts);
    m_free(workspace->sample_indices);
    m_free(workspace->feature_indices);
    m_free(workspace->min_vals);
    m_free(workspace->max_vals);
    m_free(workspace->node_stack);
    m_free(o->features_buffer);
    m_free(o->labels_buffer);

    return mp_const_none;
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_del_obj, extratrees_model_del);

// Train the model
static mp_obj_t extratrees_model_train(size_t n_args, const mp_obj_t *args) {
    // Args: self, X, y
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, X, y"));
    }
    
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlTreesModel *model = &o->model;
    EmlTreesWorkspace *workspace = &o->workspace;

    // Extract X buffer
    mp_buffer_info_t X_bufinfo;
    mp_get_buffer_raise(args[1], &X_bufinfo, MP_BUFFER_READ);
    if (X_bufinfo.typecode != 'h') {  // int16_t
        mp_raise_ValueError(MP_ERROR_TEXT("X expecting int16 array"));
    }
    const int16_t *X = X_bufinfo.buf;
    const int X_len = X_bufinfo.len / sizeof(int16_t);

    // Extract y buffer
    mp_buffer_info_t y_bufinfo;
    mp_get_buffer_raise(args[2], &y_bufinfo, MP_BUFFER_READ);
    if (y_bufinfo.typecode != 'h') {  // int16_t
        mp_raise_ValueError(MP_ERROR_TEXT("y expecting int16 array"));
    }
    const int16_t *y = y_bufinfo.buf;
    const int y_len = y_bufinfo.len / sizeof(int16_t);

    // Validate dimensions
    if (X_len != y_len * model->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("X and y dimensions don't match"));
    }

    const int16_t n_samples = y_len;
    workspace->n_samples = n_samples;

    // Copy data to internal buffers (eml_trees expects non-const pointers)
    memcpy(o->features_buffer, X, X_len * sizeof(int16_t));
    memcpy(o->labels_buffer, y, y_len * sizeof(int16_t));

    // Perform training
    int16_t result = eml_trees_train(model, workspace, o->features_buffer, o->labels_buffer);

    if (result != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Training failed"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_train_obj, 3, 3, extratrees_model_train);

// Predict using the model (returns class probabilities)
static mp_obj_t extratrees_model_predict_proba(mp_obj_fun_bc_t *self_obj,
        size_t n_args, size_t n_kw, mp_obj_t *args) {
    // Check number of arguments is valid
    mp_arg_check_num(n_args, n_kw, 3, 3, false);

    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlTreesModel *model = &o->model;

    // Extract features buffer pointer and verify typecode
    mp_buffer_info_t features_bufinfo;
    mp_get_buffer_raise(args[1], &features_bufinfo, MP_BUFFER_READ);
    if (features_bufinfo.typecode != 'h') {  // int16_t
        mp_raise_ValueError(MP_ERROR_TEXT("features expecting int16 array"));
    }
    const int16_t *features = features_bufinfo.buf;
    const int n_features = features_bufinfo.len / sizeof(int16_t);

    if (n_features != model->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    // Extract probabilities output buffer
    mp_buffer_info_t proba_bufinfo;
    mp_get_buffer_raise(args[2], &proba_bufinfo, MP_BUFFER_WRITE);
    if (proba_bufinfo.typecode != 'f') {  // float
        mp_raise_ValueError(MP_ERROR_TEXT("probabilities expecting float32 array"));
    }
    float *probabilities = proba_bufinfo.buf;
    const int proba_len = proba_bufinfo.len / sizeof(float);

    if (proba_len != model->n_classes) {
        mp_raise_ValueError(MP_ERROR_TEXT("Probabilities buffer size mismatch"));
    }

    // Allocate temporary votes buffer
    int16_t *votes = (int16_t *)m_malloc(sizeof(int16_t) * model->n_classes);

    // Make prediction
    int16_t predicted_class = eml_trees_predict_proba(model, features, probabilities, votes);

    // Free temporary buffer
    m_free(votes);

    return mp_obj_new_int(predicted_class);
}

// Predict using the model (returns only class label)
static mp_obj_t extratrees_model_predict(mp_obj_fun_bc_t *self_obj,
        size_t n_args, size_t n_kw, mp_obj_t *args) {
    // Check number of arguments is valid
    mp_arg_check_num(n_args, n_kw, 2, 2, false);

    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlTreesModel *model = &o->model;

    // Extract features buffer pointer and verify typecode
    mp_buffer_info_t features_bufinfo;
    mp_get_buffer_raise(args[1], &features_bufinfo, MP_BUFFER_READ);
    if (features_bufinfo.typecode != 'h') {  // int16_t
        mp_raise_ValueError(MP_ERROR_TEXT("features expecting int16 array"));
    }
    const int16_t *features = features_bufinfo.buf;
    const int n_features = features_bufinfo.len / sizeof(int16_t);

    if (n_features != model->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Feature count mismatch"));
    }

    // Allocate temporary buffers
    float *probabilities = (float *)m_malloc(sizeof(float) * model->n_classes);
    int16_t *votes = (int16_t *)m_malloc(sizeof(int16_t) * model->n_classes);

    // Make prediction
    int16_t predicted_class = eml_trees_predict_proba(model, features, probabilities, votes);

    // Free temporary buffers
    m_free(probabilities);
    m_free(votes);

    return mp_obj_new_int(predicted_class);
}

// Get number of features
static mp_obj_t extratrees_model_get_n_features(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_features);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_features_obj, extratrees_model_get_n_features);

// Get number of classes
static mp_obj_t extratrees_model_get_n_classes(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_classes);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_classes_obj, extratrees_model_get_n_classes);

// Get number of trees
static mp_obj_t extratrees_model_get_n_trees(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_trees);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_trees_obj, extratrees_model_get_n_trees);

// Get number of nodes used
static mp_obj_t extratrees_model_get_n_nodes_used(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_nodes_used);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_nodes_used_obj, extratrees_model_get_n_nodes_used);

// Module setup
mp_map_elem_t extratrees_model_locals_dict_table[10];
static MP_DEFINE_CONST_DICT(extratrees_model_locals_dict, extratrees_model_locals_dict_table);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&extratrees_model_new_obj));

    extratrees_model_type.base.type = (void*)&mp_fun_table.type_type;
    extratrees_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    extratrees_model_type.name = MP_QSTR_extratrees;
    
    // methods
    extratrees_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_DYNRUNTIME_MAKE_FUNCTION(extratrees_model_predict) };
    extratrees_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict_proba), MP_DYNRUNTIME_MAKE_FUNCTION(extratrees_model_predict_proba) };
    extratrees_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_train), MP_OBJ_FROM_PTR(&extratrees_model_train_obj) };
    extratrees_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&extratrees_model_del_obj) };
    extratrees_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_features), MP_OBJ_FROM_PTR(&extratrees_model_get_n_features_obj) };
    extratrees_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_classes), MP_OBJ_FROM_PTR(&extratrees_model_get_n_classes_obj) };
    extratrees_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_trees), MP_OBJ_FROM_PTR(&extratrees_model_get_n_trees_obj) };
    extratrees_model_locals_dict_table[7] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_nodes_used), MP_OBJ_FROM_PTR(&extratrees_model_get_n_nodes_used_obj) };

    MP_OBJ_TYPE_SET_SLOT(&extratrees_model_type, locals_dict, (void*)&extratrees_model_locals_dict, 8);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}
