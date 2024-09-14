// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#define EML_TREES_REGRESSION_ENABLE 0
#include <eml_trees.h>

#include <string.h>

#define EMLEARN_MICROPYTHON_DEBUG 0

// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif


// For building up an EmlTrees structure
typedef struct _EmlTreesBuilder {
    EmlTrees trees;
    int max_nodes;
    int max_trees;
    int max_leaves;
} EmlTreesBuilder;

// MicroPython type for EmlTreesBuilder
typedef struct _mp_obj_trees_builder_t {
    mp_obj_base_t base;
    EmlTreesBuilder builder;
} mp_obj_trees_builder_t;

mp_obj_full_type_t trees_builder_type;

// Create a new tree builder
static mp_obj_t builder_new(mp_obj_t trees_obj, mp_obj_t nodes_obj, mp_obj_t leaves_obj) {

    mp_int_t max_nodes = mp_obj_get_int(nodes_obj);
    mp_int_t max_trees = mp_obj_get_int(trees_obj);
    mp_int_t max_leaves = mp_obj_get_int(leaves_obj);

#if EMLEARN_MICROPYTHON_DEBUG
    mp_printf(&mp_plat_print, "builder-new nodes=%d trees=%d\n", max_nodes, max_trees);
#endif

    // create builder
    mp_obj_trees_builder_t *o = mp_obj_malloc(mp_obj_trees_builder_t, (mp_obj_type_t *)&trees_builder_type);

    EmlTreesBuilder *self = &o->builder;

    memset(self, 1, sizeof(EmlTreesBuilder)); // HACK: try to get memset symbol in

    self->max_nodes = max_nodes;
    self->max_trees = max_trees;
    self->max_leaves = max_leaves;

    // create storage for trees
    EmlTreesNode *nodes = (EmlTreesNode *)m_malloc(sizeof(EmlTreesNode)*max_nodes);
    int32_t *roots = (int32_t *)m_malloc(sizeof(int32_t)*max_trees);
    uint8_t *leaves = (uint8_t *)m_malloc(sizeof(uint8_t)*max_leaves);

#if EMLEARN_MICROPYTHON_DEBUG
    mp_printf(&mp_plat_print, "emltrees nodes=%p roots=%p builder=%p\n", nodes, roots, self);
#endif

    self->trees.n_nodes = 0;
    self->trees.nodes = nodes;

    self->trees.n_trees = 0;
    self->trees.tree_roots = roots;

    self->trees.leaf_bits = 0; // XXX: only class supported so far 
    self->trees.n_leaves = 0;
    self->trees.leaves = leaves;

    // NOTE: these are set later, in setdata()
    self->trees.n_classes = 0;
    self->trees.n_features = 0;

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_3(builder_new_obj, builder_new);

// Delete a tree builder
static mp_obj_t builder_del(mp_obj_t trees_obj) {

    mp_obj_trees_builder_t *o = MP_OBJ_TO_PTR(trees_obj);
    EmlTreesBuilder *self = &o->builder;

    // free allocated data
    m_free(self->trees.nodes);
    m_free(self->trees.tree_roots);
    m_free(self->trees.leaves);

#if EMLEARN_MICROPYTHON_DEBUG
    mp_printf(&mp_plat_print, "emltrees del \n");
#endif

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(builder_del_obj, builder_del);

// set number of features and classes
static mp_obj_t builder_setdata(mp_obj_t self_obj, mp_obj_t features_obj, mp_obj_t classes_obj) {

    mp_obj_trees_builder_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesBuilder *self = &o->builder;    

    self->trees.n_features = mp_obj_get_int(features_obj);
    self->trees.n_classes = mp_obj_get_int(classes_obj);

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_3(builder_setdata_obj, builder_setdata);


// Add a node to the tree
static mp_obj_t builder_addnode(size_t n_args, const mp_obj_t *args) {

    mp_obj_trees_builder_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlTreesBuilder *self = &o->builder;    

    const int16_t left = mp_obj_get_int(args[1]);
    const int16_t right = mp_obj_get_int(args[2]);
    const int feature = mp_obj_get_int(args[3]);
    const int16_t value = mp_obj_get_int(args[4]);

    if (feature > 127 || feature < -1) {
        mp_raise_ValueError(MP_ERROR_TEXT("feature out of bounds"));
    }

    if (self->trees.n_nodes >= self->max_nodes) {
        mp_raise_ValueError(MP_ERROR_TEXT("max nodes"));
    }

    const int node_index = self->trees.n_nodes++;
    self->trees.nodes[node_index] = (EmlTreesNode){ (int8_t)feature, value, left, right };

#if EMLEARN_MICROPYTHON_DEBUG
    mp_printf(&mp_plat_print,
        "emltrees-addnode feature=%d threshold=%f left=%d right=%d \n",
        feature, value, left, right
    );
#endif

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(builder_addnode_obj, 5, 5, builder_addnode);


// Add a node to the tree
static mp_obj_t builder_addroot(size_t n_args, const mp_obj_t *args) {

    mp_obj_trees_builder_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlTreesBuilder *self = &o->builder;    

    const int16_t root = mp_obj_get_int(args[1]);

    if (self->trees.n_trees >= self->max_trees) {
        mp_raise_ValueError(MP_ERROR_TEXT("max trees"));
    }

    const int root_index = self->trees.n_trees++;
    self->trees.tree_roots[root_index] = root; 

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(builder_addroot_obj, 2, 2, builder_addroot);


// Add a node to the tree
static mp_obj_t builder_addleaf(mp_obj_t self_obj, mp_obj_t leaf_obj) {

    mp_obj_trees_builder_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesBuilder *self = &o->builder;    

    mp_int_t leaf_value = mp_obj_get_int(leaf_obj);

    if (self->trees.n_leaves >= self->max_leaves) {
        mp_raise_ValueError(MP_ERROR_TEXT("max leaves"));
    }

    const int leaf_index = self->trees.n_leaves++;
    self->trees.leaves[leaf_index] = (uint8_t)leaf_value;

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_2(builder_addleaf_obj, builder_addleaf);


// Takes a array of input data
static mp_obj_t builder_predict(mp_obj_t self_obj, mp_obj_t features_obj) {

    mp_obj_trees_builder_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlTreesBuilder *self = &o->builder;    

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(features_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting int16 (h) array"));
    }

    const int16_t *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(*features);

#if EMLEARN_MICROPYTHON_DEBUG
    mp_printf(&mp_plat_print,
        "emltrees-predict n_features=%d n_classes=%d leaves=%d nodes=%d trees=%d length=%d \n",
        self->trees.n_features, self->trees.n_classes,
        self->trees.n_leaves, self->trees.n_nodes, self->trees.n_trees,
        n_features
    );
#endif

    // call model
    const int result = eml_trees_predict(&self->trees, features, n_features);
    if (result < 0) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("eml_trees_predict error"));
    }

    return mp_obj_new_int(result);
}
static MP_DEFINE_CONST_FUN_OBJ_2(builder_predict_obj, builder_predict);


mp_map_elem_t trees_locals_dict_table[6];
static MP_DEFINE_CONST_DICT(trees_locals_dict, trees_locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&builder_new_obj));

    trees_builder_type.base.type = (void*)&mp_fun_table.type_type;
    trees_builder_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    trees_builder_type.name = MP_QSTR_emltrees;
    // methods
    trees_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&builder_predict_obj) };
    trees_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_addnode), MP_OBJ_FROM_PTR(&builder_addnode_obj) };
    trees_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_addroot), MP_OBJ_FROM_PTR(&builder_addroot_obj) };
    trees_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_addleaf), MP_OBJ_FROM_PTR(&builder_addleaf_obj) };
    trees_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&builder_del_obj) };
    trees_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_setdata), MP_OBJ_FROM_PTR(&builder_setdata_obj) };

    MP_OBJ_TYPE_SET_SLOT(&trees_builder_type, locals_dict, (void*)&trees_locals_dict, 6);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

