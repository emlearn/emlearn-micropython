// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#define EML_LOG_ENABLE 0
#if EML_LOG_ENABLE
#define EML_LOG_PRINTF(...) mp_printf(&mp_plat_print, __VA_ARGS__)
#endif
#define EML_NEIGHBORS_LOG_LEVEL 3

#include <eml_neighbors.h>

#include <string.h>

// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif




// MicroPython type for EmlNeighborsModel
typedef struct _mp_obj_neighbors_model_t {
    mp_obj_base_t base;
    EmlNeighborsModel model;
    EmlNeighborsDistanceItem *distances;
} mp_obj_neighbors_model_t;

STATIC const mp_obj_type_t neighbors_model_type;

// Create a new instace
STATIC mp_obj_t neighbors_model_new(mp_obj_t items_obj, mp_obj_t features_obj, mp_obj_t neighbors_obj) {

    mp_int_t max_items = mp_obj_get_int(items_obj);
    mp_int_t n_features = mp_obj_get_int(features_obj);
    mp_int_t k_neighbors = mp_obj_get_int(neighbors_obj);

    // allocate space
    mp_obj_neighbors_model_t *o = \
        mp_obj_malloc(mp_obj_neighbors_model_t, (mp_obj_type_t *)&neighbors_model_type);

    EmlNeighborsModel *self = &o->model;
    memset(self, 0, sizeof(EmlNeighborsModel)); // HACK: try to get memset symbol in

    // configure model
    self->n_features = n_features;
    self->n_items = 0;
    self->max_items = max_items;
    self->data = (int16_t *)m_malloc(sizeof(int16_t)*n_features*max_items);
    self->labels = (int16_t *)m_malloc(sizeof(int16_t)*max_items);
    self->k_neighbors = k_neighbors;
    o->distances = (EmlNeighborsDistanceItem *)m_malloc(sizeof(EmlNeighborsDistanceItem)*max_items);

    return MP_OBJ_FROM_PTR(o);


}
// Define a Python reference to the function above
STATIC MP_DEFINE_CONST_FUN_OBJ_3(neighbors_model_new_obj, neighbors_model_new);


// FIXME: function for freeing a builder


// Add data to the model
STATIC mp_obj_t neighbors_model_additem(size_t n_args, const mp_obj_t *args) {

    mp_obj_neighbors_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlNeighborsModel *self = &o->model;    

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting int16 array"));
    }
    const int16_t *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(*features);

    const int16_t label = mp_obj_get_int(args[2]);

    mp_printf(&mp_plat_print, "neighbors-model-additem features=%d label=%d\n", n_features, label);
    EmlError err = eml_neighbors_add_item(self, features, n_features, label);
    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("additem failed"));
    }

    return mp_const_none;
 }
STATIC MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(neighbors_model_additem_obj, 3, 3, neighbors_model_additem);



// Takes a integer array
STATIC mp_obj_t neighbors_predict(mp_obj_fun_bc_t *self_obj,
        size_t n_args, size_t n_kw, mp_obj_t *args) {
    // Check number of arguments is valid
    mp_arg_check_num(n_args, n_kw, 2, 2, false);

    mp_obj_neighbors_model_t *o = MP_OBJ_TO_PTR(args[0]);
    EmlNeighborsModel *self = &o->model;    

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting int16 array"));
    }

    const int16_t *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(*features);

    mp_printf(&mp_plat_print, "neighbors-model-predict features=%d items=%d\n",
        n_features, self->n_items);
    // call model
    int16_t out = -1;
    const EmlError err = eml_neighbors_predict(self,
            features, n_features,
            o->distances, self->max_items,
            &out);
    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("EmlError"));
    }

    return mp_obj_new_int(out);
}


// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_open, MP_OBJ_FROM_PTR(&neighbors_model_new_obj));
    mp_store_global(MP_QSTR_predict, MP_DYNRUNTIME_MAKE_FUNCTION(neighbors_predict));
    mp_store_global(MP_QSTR_additem, MP_OBJ_FROM_PTR(&neighbors_model_additem_obj));


    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

