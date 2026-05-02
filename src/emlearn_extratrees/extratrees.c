// Include the header file to get access to the MicroPython API
#ifdef MICROPY_ENABLE_DYNRUNTIME
#include "py/dynruntime.h"
#else
#include "py/runtime.h"
#endif

#include <string.h>

#include "eml_extratrees.c"

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

// MicroPython type for ExtraTrees model
typedef struct _mp_obj_extratrees_model_t {
    mp_obj_base_t base;
    EmlExtraTreesModel model;
    EmlExtraTreesWorkspace workspace;
    mp_obj_t train_X_obj;  // Reference to X Python object (prevents GC during step training)
    mp_obj_t train_y_obj;  // Reference to y Python object (prevents GC during step training)
} mp_obj_extratrees_model_t;

#if MICROPY_ENABLE_DYNRUNTIME
mp_obj_full_type_t extratrees_model_type;
#else
static const mp_obj_type_t extratrees_model_type;
#endif

// Create a new instance
static mp_obj_t extratrees_model_new(size_t n_args, const mp_obj_t *args) {
    // Args: n_features, n_classes, [n_trees], [max_depth], [min_samples_leaf], [n_thresholds], 
    //       [subsample_ratio], [feature_subsample_ratio], [max_nodes], [max_samples], [rng_seed], [use_global_feature_range]
    if (n_args < 2 || n_args > 12) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 2-12 arguments: n_features, n_classes, [n_trees=10], [max_depth=10], [min_samples_leaf=1], [n_thresholds=10], [subsample_ratio=1.0], [feature_subsample_ratio=1.0], [max_nodes=1000], [max_samples=1000], [rng_seed=42], [use_global_feature_range=0]"));
    }
    
    mp_int_t n_features = mp_obj_get_int(args[0]);
    mp_int_t n_classes = mp_obj_get_int(args[1]);
    mp_int_t n_trees = (n_args > 2) ? mp_obj_get_int(args[2]) : 10;
    mp_int_t max_depth = (n_args > 3) ? mp_obj_get_int(args[3]) : 10;
    mp_int_t min_samples_leaf = (n_args > 4) ? mp_obj_get_int(args[4]) : 1;
    mp_int_t n_thresholds = (n_args > 5) ? mp_obj_get_int(args[5]) : 10;
    float subsample_ratio = (n_args > 6) ? mp_obj_get_float_to_f(args[6]) : 1.0f;
    float feature_subsample_ratio = (n_args > 7) ? mp_obj_get_float_to_f(args[7]) : 1.0f;
    mp_int_t max_nodes = (n_args > 8) ? mp_obj_get_int(args[8]) : 1000;
    mp_int_t max_samples = (n_args > 9) ? mp_obj_get_int(args[9]) : 1000;
    mp_int_t rng_seed = (n_args > 10) ? mp_obj_get_int(args[10]) : 42;
    mp_int_t use_global_feature_range = (n_args > 11) ? mp_obj_get_int(args[11]) : 0;

    // Allocate space
    mp_obj_extratrees_model_t *o = \
        mp_obj_malloc(mp_obj_extratrees_model_t, (mp_obj_type_t *)&extratrees_model_type);

    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;
    memset(model, 0, sizeof(EmlExtraTreesModel));
    memset(workspace, 0, sizeof(EmlExtraTreesWorkspace));

    // Configure model
    model->n_features = n_features;
    model->n_classes = n_classes;
    model->n_trees = n_trees;
    model->max_nodes = max_nodes;
    model->max_samples = max_samples;
    model->n_nodes_used = 0;
    model->n_trees_trained = 0;
    
    // Configure model config
    model->config.max_depth = max_depth;
    model->config.min_samples_leaf = min_samples_leaf;
    model->config.n_thresholds = n_thresholds;
    model->config.subsample_ratio = subsample_ratio;
    model->config.feature_subsample_ratio = feature_subsample_ratio;
    model->config.rng_seed = rng_seed;
    model->config.use_global_feature_range = use_global_feature_range;
    
    // Allocate model buffers
    model->nodes = m_new(EmlExtraTreesNode, max_nodes);
    model->tree_starts = m_new(int16_t, n_trees);
    
    // Allocate workspace buffers
    workspace->sample_indices = m_new(uint16_t, max_samples);
    workspace->feature_indices = m_new(uint16_t, n_features);
    workspace->min_vals = m_new(int16_t, n_features);
    workspace->max_vals = m_new(int16_t, n_features);
    workspace->class_counts = m_new(int16_t, n_classes);
    // Allocate temporary arrays for find_best_split
    workspace->split_left_counts = m_new(int16_t, n_classes);
    workspace->split_right_counts = m_new(int16_t, n_classes);
    // Allocate stack with enough capacity for max depth (conservative estimate)
    workspace->node_stack = m_new(NodeState, max_depth * 3);
    // Allocate temporary arrays for predict
    workspace->probabilities = m_new(float, n_classes);
    workspace->votes = m_new(int16_t, n_classes);
    workspace->n_samples = 0; // Will be set during training
    workspace->rng_state = rng_seed;
    
    // No training data buffers - we use Python buffer pointers directly
    o->train_X_obj = mp_const_none;
    o->train_y_obj = mp_const_none;
    
    // Initialize nodes and tree starts
    memset(model->nodes, 0, sizeof(EmlExtraTreesNode) * max_nodes);
    memset(model->tree_starts, 0, sizeof(int16_t) * n_trees);

    return MP_OBJ_FROM_PTR(o);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_new_obj, 2, 12, extratrees_model_new);

// Delete an instance
static mp_obj_t extratrees_model_del(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;

    // Free allocated memory
    m_del(EmlExtraTreesNode, model->nodes, model->max_nodes);
    m_del(int16_t, model->tree_starts, model->n_trees);
    m_del(uint16_t, workspace->sample_indices, model->max_samples);
    m_del(uint16_t, workspace->feature_indices, model->n_features);
    m_del(int16_t, workspace->min_vals, model->n_features);
    m_del(int16_t, workspace->max_vals, model->n_features);
    m_del(int16_t, workspace->class_counts, model->n_classes);
    m_del(int16_t, workspace->split_left_counts, model->n_classes);
    m_del(int16_t, workspace->split_right_counts, model->n_classes);
    m_del(NodeState, workspace->node_stack, model->config.max_depth * 3);
    m_del(float, workspace->probabilities, model->n_classes);
    m_del(int16_t, workspace->votes, model->n_classes);

    // Release training data Python object references
    o->train_X_obj = mp_const_none;
    o->train_y_obj = mp_const_none;

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
    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;

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

    // Pass buffer pointers directly (no copy needed)
    int16_t result = eml_extratrees_train(model, workspace, X, y);

    if (result != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Training failed"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_train_obj, 3, 3, extratrees_model_train);

// Initialize step-by-step training
static mp_obj_t extratrees_model_train_init(size_t n_args, const mp_obj_t *args) {
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, X, y"));
    }
    
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;

    // Extract X buffer
    mp_buffer_info_t X_bufinfo;
    mp_get_buffer_raise(args[1], &X_bufinfo, MP_BUFFER_READ);
    if (X_bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("X expecting int16 array"));
    }
    const int16_t *X = X_bufinfo.buf;
    const int X_len = X_bufinfo.len / sizeof(int16_t);

    // Extract y buffer
    mp_buffer_info_t y_bufinfo;
    mp_get_buffer_raise(args[2], &y_bufinfo, MP_BUFFER_READ);
    if (y_bufinfo.typecode != 'h') {
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

    // Hold Python object references to prevent GC during step-by-step training
    o->train_X_obj = args[1];
    o->train_y_obj = args[2];

    // Pass buffer pointers directly (no copy needed)
    int16_t result = eml_extratrees_train_init(model, workspace, X, y);
    if (result != 0) {
        o->train_X_obj = mp_const_none;
        o->train_y_obj = mp_const_none;
        mp_raise_ValueError(MP_ERROR_TEXT("Training init failed"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_train_init_obj, 3, 3, extratrees_model_train_init);

// Process one node in step-by-step training
// Returns: 1=training complete, 0=more steps needed
static mp_obj_t extratrees_model_train_step(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;

    int16_t result = eml_extratrees_train_step(model, workspace);
    if (result < 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("Train step failed"));
    }

    // Release Python object references when training is complete
    if (result == 1) {
        o->train_X_obj = mp_const_none;
        o->train_y_obj = mp_const_none;
    }

    return mp_obj_new_int(result);
}
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_train_step_obj, extratrees_model_train_step);

// Predict using the model (returns class probabilities)
static mp_obj_t extratrees_model_predict_proba(size_t n_args, const mp_obj_t *args) {
    if (n_args != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 3 arguments: self, features, probabilities"));
    }

    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;

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

    // Make prediction using pre-allocated workspace arrays
    int16_t predicted_class = eml_extratrees_predict_proba(model, features, probabilities, workspace->votes);

    return mp_obj_new_int(predicted_class);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_predict_proba_obj, 3, 3, extratrees_model_predict_proba);

// Predict using the model (returns only class label)
static mp_obj_t extratrees_model_predict(size_t n_args, const mp_obj_t *args) {
    if (n_args != 2) {
        mp_raise_ValueError(MP_ERROR_TEXT("Expected 2 arguments: self, features"));
    }

    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlExtraTreesModel *model = &o->model;
    EmlExtraTreesWorkspace *workspace = &o->workspace;

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

    // Make prediction using pre-allocated workspace arrays
    int16_t predicted_class = eml_extratrees_predict_proba(model, features, workspace->probabilities, workspace->votes);

    return mp_obj_new_int(predicted_class);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(extratrees_model_predict_obj, 2, 2, extratrees_model_predict);

// Get number of features
static mp_obj_t extratrees_model_get_n_features(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_features);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_features_obj, extratrees_model_get_n_features);

// Get number of classes
static mp_obj_t extratrees_model_get_n_classes(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_classes);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_classes_obj, extratrees_model_get_n_classes);

// Get number of trees
static mp_obj_t extratrees_model_get_n_trees(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_trees);
}
// Define a Python reference to the function above
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_trees_obj, extratrees_model_get_n_trees);

// Get number of nodes used
static mp_obj_t extratrees_model_get_n_nodes_used(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_nodes_used);
}
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_nodes_used_obj, extratrees_model_get_n_nodes_used);

// Get number of trees trained
static mp_obj_t extratrees_model_get_n_trees_trained(mp_obj_t self_obj) {
    mp_obj_extratrees_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlExtraTreesModel *model = &o->model;

    return mp_obj_new_int(model->n_trees_trained);
}
static MP_DEFINE_CONST_FUN_OBJ_1(extratrees_model_get_n_trees_trained_obj, extratrees_model_get_n_trees_trained);

// =========================================================================
// Module registration — dynruntime vs user C module
// =========================================================================

#if MICROPY_ENABLE_DYNRUNTIME

// Module setup
mp_map_elem_t extratrees_model_locals_dict_table[11];
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
    extratrees_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&extratrees_model_predict_obj) };
    extratrees_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict_proba), MP_OBJ_FROM_PTR(&extratrees_model_predict_proba_obj) };
    extratrees_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_train), MP_OBJ_FROM_PTR(&extratrees_model_train_obj) };
    extratrees_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_train_init), MP_OBJ_FROM_PTR(&extratrees_model_train_init_obj) };
    extratrees_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_train_step), MP_OBJ_FROM_PTR(&extratrees_model_train_step_obj) };
    extratrees_model_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&extratrees_model_del_obj) };
    extratrees_model_locals_dict_table[6] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_features), MP_OBJ_FROM_PTR(&extratrees_model_get_n_features_obj) };
    extratrees_model_locals_dict_table[7] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_classes), MP_OBJ_FROM_PTR(&extratrees_model_get_n_classes_obj) };
    extratrees_model_locals_dict_table[8] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_trees), MP_OBJ_FROM_PTR(&extratrees_model_get_n_trees_obj) };
    extratrees_model_locals_dict_table[9] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_nodes_used), MP_OBJ_FROM_PTR(&extratrees_model_get_n_nodes_used_obj) };
    extratrees_model_locals_dict_table[10] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_get_n_trees_trained), MP_OBJ_FROM_PTR(&extratrees_model_get_n_trees_trained_obj) };

    MP_OBJ_TYPE_SET_SLOT(&extratrees_model_type, locals_dict, (void*)&extratrees_model_locals_dict, 10);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

#else

// User C module mode
static const mp_rom_map_elem_t extratrees_model_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&extratrees_model_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict_proba), MP_ROM_PTR(&extratrees_model_predict_proba_obj) },
    { MP_ROM_QSTR(MP_QSTR_train), MP_ROM_PTR(&extratrees_model_train_obj) },
    { MP_ROM_QSTR(MP_QSTR_train_init), MP_ROM_PTR(&extratrees_model_train_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_train_step), MP_ROM_PTR(&extratrees_model_train_step_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&extratrees_model_del_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_features), MP_ROM_PTR(&extratrees_model_get_n_features_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_classes), MP_ROM_PTR(&extratrees_model_get_n_classes_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_trees), MP_ROM_PTR(&extratrees_model_get_n_trees_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_nodes_used), MP_ROM_PTR(&extratrees_model_get_n_nodes_used_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_n_trees_trained), MP_ROM_PTR(&extratrees_model_get_n_trees_trained_obj) },
};
static MP_DEFINE_CONST_DICT(extratrees_model_locals_dict, extratrees_model_locals_dict_table);

static MP_DEFINE_CONST_OBJ_TYPE(
    extratrees_model_type,
    MP_QSTR_extratrees,
    MP_TYPE_FLAG_ITER_IS_CUSTOM,
    locals_dict, &extratrees_model_locals_dict
);

// Define module object
static const mp_rom_map_elem_t emlearn_extratrees_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_new), MP_ROM_PTR(&extratrees_model_new_obj) },
};
static MP_DEFINE_CONST_DICT(emlearn_extratrees_globals, emlearn_extratrees_globals_table);

const mp_obj_module_t emlearn_extratrees_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&emlearn_extratrees_globals,
};

MP_REGISTER_MODULE(MP_QSTR_emlearn_extratrees_c, emlearn_extratrees_cmodule);

#endif
