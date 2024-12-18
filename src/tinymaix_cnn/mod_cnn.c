// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

// TinyMaix config
#include "./tm_port.h"

#include <tinymaix.h>

#include "tm_layers.c"
#include "tm_model.c"
//#include "tm_stat.c"

#include <string.h>


// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}

int
__aeabi_idiv0(int return_value) {
  return return_value;
}

void NORETURN abort() {
    while (1) {
        ;
    }
}
#endif

// get model output shapes
//mdl: model handle; in: input mat; out: output mat
int TM_WEAK tm_get_outputs(tm_mdl_t* mdl, tm_mat_t* out, int out_length)
{
    // NOTE: based on tm_run, but without actually executing
    int out_idx = 0;
    mdl->layer_body = mdl->b->layers_body;
    for(mdl->layer_i = 0; mdl->layer_i < mdl->b->layer_cnt; mdl->layer_i++){
        tml_head_t* h = (tml_head_t*)(mdl->layer_body);
        if(h->is_out) {
            if (out_idx < out_length) {
                memcpy((void*)(&out[out_idx]), (void*)(&(h->out_dims)), sizeof(uint16_t)*4);
                out_idx += 1;
            } else {
                return -1;
            }
        }
        mdl->layer_body += (h->size);
    }
    return out_idx;
}

static tm_err_t layer_cb(tm_mdl_t* mdl, tml_head_t* lh)
{
    return TM_OK;
}

#define DEBUG (1)

// MicroPython type
typedef struct _mp_obj_mod_cnn_t {
    mp_obj_base_t base;

    tm_mdl_t model;
    tm_mat_t input;
    uint8_t *model_buffer;
    uint8_t *data_buffer;
    uint16_t out_dims[4];
} mp_obj_mod_cnn_t;

mp_obj_full_type_t mod_cnn_type;

// TODO: add function for getting the shape of expected input. As a tuple

// Create a new instance
static mp_obj_t mod_cnn_new(mp_obj_t model_data_obj) {

    // Check model data
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(model_data_obj, &bufinfo, MP_BUFFER_RW);

#if DEBUG
    mp_printf(&mp_plat_print, "cnn-new data.typecode=%c \n", bufinfo.typecode);
#endif

    if (bufinfo.typecode != 'B') {
        mp_raise_ValueError(MP_ERROR_TEXT("model should be bytes"));
    }
    uint8_t *model_data_buffer = bufinfo.buf;
    const int model_data_length = bufinfo.len / sizeof(*model_data_buffer);

    // Construct object
    mp_obj_mod_cnn_t *o = mp_obj_malloc(mp_obj_mod_cnn_t, (mp_obj_type_t *)&mod_cnn_type);
    tm_mdl_t *model = &o->model;

    // Copy the model data
    o->model_buffer = m_malloc(model_data_length);
    memcpy(o->model_buffer, model_data_buffer, model_data_length);

    // Allocate temporary buffer
    // TODO: this can possibly be smaller? Might want to use TinyMaix internal alloc
    o->data_buffer = m_malloc(model_data_length);

    // loading model
    // will set the dimensions of the input matrix
    tm_err_t load_err = tm_load(model, o->model_buffer, o->data_buffer, layer_cb, &o->input);
    if (load_err != TM_OK) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("tm_load error"));
    }

    // find model output shape
    o->out_dims[0] = 0;
    tm_mat_t outs[1];
    const int outputs = tm_get_outputs(model, outs, 1);
    if (outputs != 1) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("only 1 output supported"));
    }
    memcpy((void*)(o->out_dims), (void*)(&(outs[0])), sizeof(uint16_t)*4);

    if ((o->out_dims[0] != 1)) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("output must be 1d"));
    }
    memcpy((void*)(o->out_dims), (void*)(&(outs[0])), sizeof(uint16_t)*4);

#if DEBUG
    mp_printf(&mp_plat_print, "cnn-new-done outs=%d out.dims=(%d,%d,%d,%d) \n",
        outputs, o->out_dims[0], o->out_dims[1], o->out_dims[2], o->out_dims[3]);
#endif

    return MP_OBJ_FROM_PTR(o);
}
static MP_DEFINE_CONST_FUN_OBJ_1(mod_cnn_new_obj, mod_cnn_new);

// Delete the instance
static mp_obj_t mod_cnn_del(mp_obj_t self_obj) {

    mp_obj_mod_cnn_t *o = MP_OBJ_TO_PTR(self_obj);
    tm_mdl_t *model = &o->model;

    m_free(o->model_buffer);
    m_free(o->data_buffer);
    tm_unload(model);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(mod_cnn_del_obj, mod_cnn_del);


// Add a node to the tree
static mp_obj_t mod_cnn_run(mp_obj_t self_obj, mp_obj_t input_obj, mp_obj_t output_obj) {

    mp_obj_mod_cnn_t *o = MP_OBJ_TO_PTR(self_obj);

    // Extract input
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(input_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'B') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting byte array"));
    }
    uint8_t *input_buffer = bufinfo.buf;
    const int input_length = bufinfo.len / sizeof(*input_buffer);

    // check buffer size wrt input
    const int expect_length = o->input.h * o->input.w * o->input.c;
    if (input_length != expect_length) {
        mp_raise_ValueError(MP_ERROR_TEXT("wrong input size"));
    }

    // Extract output
    mp_get_buffer_raise(output_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
    }
    float *output_buffer = bufinfo.buf;
    const int output_length = bufinfo.len / sizeof(*output_buffer);


    // check buffer size wrt input
    const int expect_out_length = o->out_dims[1]*o->out_dims[2]*o->out_dims[3];
    if (output_length != expect_out_length) {
        mp_raise_ValueError(MP_ERROR_TEXT("wrong output size"));
    }

    // Preprocess data
    tm_mat_t in_uint8 = o->input;
    in_uint8.data = (mtype_t*)input_buffer;

#if (TM_MDL_TYPE == TM_MDL_INT8) || (TM_MDL_TYPE == TM_MDL_INT16) 
    const tm_err_t preprocess_err = tm_preprocess(&o->model, TMPP_UINT2INT, &in_uint8, &o->input); 
#else
    const tm_err_t preprocess_err = tm_preprocess(&o->model, TMPP_UINT2FP01, &in_uint8, &o->input); 
#endif
    if (preprocess_err != TM_OK) {
        mp_raise_ValueError(MP_ERROR_TEXT("preprocess error"));
    }

    // Run the CNN
    tm_mat_t outs[1];
    tm_err_t run_err = tm_run(&o->model, &o->input, outs);

    if (run_err != TM_OK) {
        mp_raise_ValueError(MP_ERROR_TEXT("run error"));
    }

    // Copy output into
    tm_mat_t out = outs[0];
    for(int i=0; i<expect_out_length; i++){
        output_buffer[i] = out.dataf[i];
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(mod_cnn_run_obj, mod_cnn_run);


// Return the shape of the output
static mp_obj_t mod_cnn_output_dimensions(mp_obj_t self_obj) {

    mp_obj_mod_cnn_t *o = MP_OBJ_TO_PTR(self_obj);
    const int dimensions = o->out_dims[0];
    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(dimensions, NULL));

    // A regular output should have C channels, and 1 for everything else
    // TODO: support other shapes?
    //dims==1, 11c
    if (!(o->out_dims[0] == 1 && o->out_dims[1] == 1 && o->out_dims[2] == 1)) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("wrong output shape"));
    }

    tuple->items[0] = mp_obj_new_int(o->out_dims[3]);
    return tuple;
}
static MP_DEFINE_CONST_FUN_OBJ_1(mod_cnn_output_dimensions_obj, mod_cnn_output_dimensions);


mp_map_elem_t mod_locals_dict_table[3];
static MP_DEFINE_CONST_DICT(mod_locals_dict, mod_locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&mod_cnn_new_obj));

    mod_cnn_type.base.type = (void*)&mp_fun_table.type_type;
    mod_cnn_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    mod_cnn_type.name = MP_QSTR_tinymaixcnn;
    // methods
    mod_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_run), MP_OBJ_FROM_PTR(&mod_cnn_run_obj) };
    mod_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&mod_cnn_del_obj) };
    mod_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_output_dimensions), MP_OBJ_FROM_PTR(&mod_cnn_output_dimensions_obj) };

    MP_OBJ_TYPE_SET_SLOT(&mod_cnn_type, locals_dict, (void*)&mod_locals_dict, 2);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

