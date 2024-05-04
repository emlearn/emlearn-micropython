// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

// TinyMaix config
#include "./tm_port.h"

#include <tinymaix.h>

#include "tm_layers.c"
#include "tm_model.c"
//#include "tm_stat.c"

#include <string.h>


static tm_err_t layer_cb(tm_mdl_t* mdl, tml_head_t* lh)
{  
#if 0
    //dump middle result
    int h = lh->out_dims[1];
    int w = lh->out_dims[2];
    int ch= lh->out_dims[3];
    mtype_t* output = TML_GET_OUTPUT(mdl, lh);
    return TM_OK;
    TM_PRINTF("Layer %d callback ========\n", mdl->layer_i);
    #if 1
    for(int y=0; y<h; y++){
        TM_PRINTF("[");
        for(int x=0; x<w; x++){
            TM_PRINTF("[");
            for(int c=0; c<ch; c++){
            #if TM_MDL_TYPE == TM_MDL_FP32
                TM_PRINTF("%.3f,", output[(y*w+x)*ch+c]);
            #else
                TM_PRINTF("%.3f,", TML_DEQUANT(lh,output[(y*w+x)*ch+c]));
            #endif
            }
            TM_PRINTF("],");
        }
        TM_PRINTF("],\n");
    }
    TM_PRINTF("\n");
    #endif
    return TM_OK;
#else
    return TM_OK;
#endif
}

#define DEBUG (1)

// MicroPython type
typedef struct _mp_obj_mod_cnn_t {
    mp_obj_base_t base;

    tm_mdl_t model;
    tm_mat_t input;
    uint8_t *model_buffer;
    uint8_t *data_buffer;
} mp_obj_mod_cnn_t;

mp_obj_full_type_t mod_cnn_type;

// TODO: add function for getting the shape of expected input. As a tuple

// Create a new instance
STATIC mp_obj_t mod_cnn_new(mp_obj_t model_data_obj) {

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
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("eml_fft_forward error"));
    }

    return MP_OBJ_FROM_PTR(o);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(mod_cnn_new_obj, mod_cnn_new);

// Delete the instance
STATIC mp_obj_t mod_cnn_del(mp_obj_t self_obj) {

    mp_obj_mod_cnn_t *o = MP_OBJ_TO_PTR(self_obj);
    tm_mdl_t *model = &o->model;

    m_free(o->model_buffer);
    m_free(o->data_buffer);
    tm_unload(model);

    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(mod_cnn_del_obj, mod_cnn_del);


// Add a node to the tree
STATIC mp_obj_t mod_cnn_run(mp_obj_t self_obj, mp_obj_t input_obj) {

    mp_obj_mod_cnn_t *o = MP_OBJ_TO_PTR(self_obj);

    // Extract input
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(input_obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'B') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
    }
    uint8_t *input_buffer = bufinfo.buf;
    const int input_length = bufinfo.len / sizeof(*input_buffer);

    // check buffer size wrt input
    const int expect_length = o->input.h * o->input.w * o->input.c;
    if (input_length != expect_length) {
        mp_raise_ValueError(MP_ERROR_TEXT("wrong input size"));
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

    tm_mat_t out = outs[0];
    float* data  = out.dataf;
    float maxp = 0;
    int maxi = -1;

    // TODO: pass the entire output vector out to Python
    // FIXME: unhardcode output handling
    for(int i=0; i<10; i++){
        //printf("%d: %.3f\n", i, data[i]);
        if (data[i] > maxp) {
            maxi = i;
            maxp = data[i];
        }
    }

    return mp_obj_new_int(maxi);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(mod_cnn_run_obj, mod_cnn_run);


mp_map_elem_t mod_locals_dict_table[2];
STATIC MP_DEFINE_CONST_DICT(mod_locals_dict, mod_locals_dict_table);

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

    MP_OBJ_TYPE_SET_SLOT(&mod_cnn_type, locals_dict, (void*)&mod_locals_dict, 2);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

