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

mp_obj_full_type_t neighbors_model_type;

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


// Delete an instance
STATIC mp_obj_t neighbors_model_del(mp_obj_t self_obj) {

    mp_obj_neighbors_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlNeighborsModel *self = &o->model;   

    // free allocated memory
    m_free(self->data);
    m_free(self->labels);
    m_free(o->distances);

    return mp_const_none;
}
// Define a Python reference to the function above
STATIC MP_DEFINE_CONST_FUN_OBJ_1(neighbors_model_del_obj, neighbors_model_del);


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

    const int item_idx = self->n_items;
    EmlError err = eml_neighbors_add_item(self, features, n_features, label);
    if (err != EmlOk) {
        mp_raise_ValueError(MP_ERROR_TEXT("additem failed"));
    }

    return mp_obj_new_int(item_idx);
 }
STATIC MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(neighbors_model_additem_obj, 3, 3, neighbors_model_additem);


// Access data of an item
STATIC mp_obj_t neighbors_model_get_item(mp_obj_t self_obj, mp_obj_t index_obj, mp_obj_t out_obj) {

    mp_obj_neighbors_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlNeighborsModel *self = &o->model;

    const mp_int_t index = mp_obj_get_int(index_obj);
    if (index < 0 || index >= self->n_items) {
        mp_raise_ValueError(MP_ERROR_TEXT("Index out of bounds"));
    }

    // Extract buffer pointer and verify typecode
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(out_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'h') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting int16 array"));
    }
    int16_t *features = bufinfo.buf;
    const int n_features = bufinfo.len / sizeof(*features);

    if (n_features != self->n_features) {
        mp_raise_ValueError(MP_ERROR_TEXT("Buffer is wrong size"));
    }

    const int16_t *item = self->data + (index*n_features);
    memcpy(features, item, sizeof(int16_t)*n_features);

    return mp_const_none;
}
// Define a Python reference to the function above
STATIC MP_DEFINE_CONST_FUN_OBJ_3(neighbors_model_get_item_obj, neighbors_model_get_item);




// Takes a integer array
STATIC mp_obj_t neighbors_model_predict(mp_obj_fun_bc_t *self_obj,
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

// Access details about prediction result
STATIC mp_obj_t neighbors_model_get_result(mp_obj_t self_obj, mp_obj_t index_obj) {

    mp_obj_neighbors_model_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlNeighborsModel *self = &o->model;

    const mp_int_t index = mp_obj_get_int(index_obj);
    if (index < 0 || index >= self->n_items) {
        mp_raise_ValueError(MP_ERROR_TEXT("Index out of bounds"));
    }

    const EmlNeighborsDistanceItem *item = &o->distances[index];

    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(3, NULL));
    tuple->items[0] = mp_obj_new_int(item->index);
    tuple->items[1] = mp_obj_new_int(item->distance);
    tuple->items[2] = mp_obj_new_int(self->labels[item->index]);

    return tuple;
}
// Define a Python reference to the function above
STATIC MP_DEFINE_CONST_FUN_OBJ_2(neighbors_model_get_result_obj, neighbors_model_get_result);




// Module setup
mp_map_elem_t neighbors_model_locals_dict_table[5];
STATIC MP_DEFINE_CONST_DICT(neighbors_model_locals_dict, neighbors_model_locals_dict_table);

// Module setup entrypoint
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&neighbors_model_new_obj));

    neighbors_model_type.base.type = (void*)&mp_fun_table.type_type;
    neighbors_model_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    neighbors_model_type.name = MP_QSTR_emlneighbors;
    // methods
    neighbors_model_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_DYNRUNTIME_MAKE_FUNCTION(neighbors_model_predict) };
    neighbors_model_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_additem), MP_OBJ_FROM_PTR(&neighbors_model_additem_obj) };
    neighbors_model_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&neighbors_model_del_obj) };
    neighbors_model_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_getresult), MP_OBJ_FROM_PTR(&neighbors_model_get_result_obj) };
    neighbors_model_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_getitem), MP_OBJ_FROM_PTR(&neighbors_model_get_item_obj) };

    MP_OBJ_TYPE_SET_SLOT(&neighbors_model_type, locals_dict, (void*)&neighbors_model_locals_dict, 5);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

