// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include <eml_common.h>

#include <string.h>

#define DEBUG (0)

// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}

void NORETURN abort() {
    while (1) {
        ;
    }
}
#endif


// Copy of eml_fft.h, without eml_fft_fill
// - contains sin/cos that trips up mpy_ld.py (even if the function is not used)
static size_t reverse_bits(size_t x, int n) {
	size_t result = 0;
	for (int i = 0; i < n; i++, x >>= 1)
		result = (result << 1) | (x & 1U);
	return result;
}

EmlError
fft_forward(float *table_sin, float *table_cos, float real[], float imag[], size_t n) {

    // Compute levels = floor(log2(n))
	int levels = 0;
	for (size_t temp = n; temp > 1U; temp >>= 1)
		levels++;

    EML_PRECONDITION(((size_t)(1U << levels)) == n, EmlSizeMismatch);

	// Bit-reversed addressing permutation
	for (size_t i = 0; i < n; i++) {
		size_t j = reverse_bits(i, levels);
		if (j > i) {
			float temp = real[i];
			real[i] = real[j];
			real[j] = temp;
			temp = imag[i];
			imag[i] = imag[j];
			imag[j] = temp;
		}
	}

	// Cooley-Tukey decimation-in-time radix-2 FFT
	for (size_t size = 2; size <= n; size *= 2) {
		size_t halfsize = size / 2;
		size_t tablestep = n / size;
		for (size_t i = 0; i < n; i += size) {
			for (size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
				size_t l = j + halfsize;
				float tpre =  real[l] * table_cos[k] + imag[l] * table_sin[k];
				float tpim = -real[l] * table_sin[k] + imag[l] * table_cos[k];
				real[l] = real[j] - tpre;
				imag[l] = imag[j] - tpim;
				real[j] += tpre;
				imag[j] += tpim;
			}
		}
		if (size == n)  // Prevent overflow in 'size *= 2'
			break;
	}	
	return EmlOk;
}



// MicroPython type for EmlFFT
mp_obj_full_type_t mp_fft_type;

typedef struct _mp_obj_fft_t {
    mp_obj_base_t base;

    int length; // (n/2)
    float *sin;
    float *cos;
    bool filled;
} mp_obj_fft_t;

// Create a new instance
static mp_obj_t fft_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args_in) {

    mp_arg_check_num(n_args, n_kw, 1, 1, false);

#if DEBUG
    mp_printf(&mp_plat_print, "fft-new-start length=%d\n",
        0);
#endif

    const int fft_length = mp_obj_get_int(args_in[0]);

    // Construct object
    mp_obj_fft_t *o = mp_obj_malloc(mp_obj_fft_t, type);
#if 1
    //EmlFFT *self = &o->fft;
    o->filled = false;

    const int table_length = fft_length / 2;
    o->cos = m_new(float, table_length);
    o->sin = m_new(float, table_length);
    o->length = table_length;

    if (o->cos == NULL || o->sin == NULL) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("allocation failed"));
    }
#endif

#if DEBUG
    mp_printf(&mp_plat_print, "fft-new length=%d\n",
        fft_length);
#endif

#if 0
    // Using sinf/cosf does not work with mpy_ld.py
    // https://github.com/micropython/micropython/issues/14430
    // As a workaround, we require the FFT factors to be filled from the MicroPython side
    const EmlError err = eml_fft_fill(self, fft_length);
    if (err != EmlOk) {
        m_free(self.cos);
        m_free(self.sin);
        mp_raise_ValueError(MP_ERROR_TEXT("EmlError"));
    }
#endif

    return MP_OBJ_FROM_PTR(o);
}

// Delete the instance
static mp_obj_t fft_del(mp_obj_t self_obj) {

#if DEBUG
    mp_printf(&mp_plat_print, "fft-del-start \n");
#endif

#if 0
    mp_obj_fft_t *o = MP_OBJ_TO_PTR(self_obj);

    EmlFFT *self = &o->fft;

    // free allocated data
    m_free(self->cos);
    m_free(self->sin);
#endif

#if DEBUG
    mp_printf(&mp_plat_print, "fft-del \n");
#endif

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(fft_del_obj, fft_del);


float *
check_extract_array(mp_obj_t obj, int length) {

    // Verify real
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(obj, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("expecting float array"));
        return NULL;
    }
    float *values = bufinfo.buf;
    const int array_length = bufinfo.len / sizeof(*values);

#if DEBUG
    mp_printf(&mp_plat_print, "check-array expect=%d got=%d\n",
        length, array_length);
#endif

    if (array_length != length) {
        mp_raise_ValueError(MP_ERROR_TEXT("wrong array length"));
        return NULL;
    }

    return values;
}

// Fill the FFT factors
static mp_obj_t fft_fill(mp_obj_t self_obj, mp_obj_t sin_obj, mp_obj_t cos_obj) {

    mp_obj_fft_t *o = MP_OBJ_TO_PTR(self_obj);
    //EmlFFT *self = &o->fft;
    const int length = o->length;


    float *sin_values = check_extract_array(sin_obj, length);
    float *cos_values = check_extract_array(cos_obj, length);

    memcpy(o->sin, sin_values, sizeof(float)*length);
    memcpy(o->cos, cos_values, sizeof(float)*length);
    o->filled = true;

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_3(fft_fill_obj, fft_fill);

// Compute the FFT
static mp_obj_t fft_run(mp_obj_t self_obj, mp_obj_t real_obj, mp_obj_t imag_obj) {

    mp_obj_fft_t *o = MP_OBJ_TO_PTR(self_obj);
    //EmlFFT *self = &o->fft;
    const int fft_length = o->length*2;

    if (!o->filled) {
        mp_raise_ValueError(MP_ERROR_TEXT("fill() not called first"));
    }

    float *real_values = check_extract_array(real_obj, fft_length);
    float *imag_values = check_extract_array(imag_obj, fft_length);

    const EmlError err = fft_forward(o->sin, o->cos, real_values, imag_values, fft_length);
    if (err != EmlOk) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("eml_fft_forward error"));
    }

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_3(fft_run_obj, fft_run);


mp_map_elem_t mod_locals_dit_table[3];
static MP_DEFINE_CONST_DICT(mod_locals_dit, mod_locals_dit_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

#if DEBUG
    mp_printf(&mp_plat_print, "fft-mpy-init\n");
#endif

    mp_fft_type.base.type = (void*)&mp_type_type;
    mp_fft_type.flags = MP_TYPE_FLAG_NONE;
    mp_fft_type.name = MP_QSTR_FFT;
    MP_OBJ_TYPE_SET_SLOT(&mp_fft_type, make_new, fft_new, 0);

    // methods
    mod_locals_dit_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_run), MP_OBJ_FROM_PTR(&fft_run_obj) };
    mod_locals_dit_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&fft_del_obj) };
    mod_locals_dit_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_fill), MP_OBJ_FROM_PTR(&fft_fill_obj) };
    MP_OBJ_TYPE_SET_SLOT(&mp_fft_type, locals_dict, (void*)&mod_locals_dit, 3);

    // Make the Factorial type available on the module.
    mp_store_global(MP_QSTR_FFT, MP_OBJ_FROM_PTR(&mp_fft_type));

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

