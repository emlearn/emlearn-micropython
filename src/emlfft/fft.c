// Include the header file to get access to the MicroPython API
#include "py/dynruntime.h"

#include <eml_common.h>

#include <string.h>
#include <errno.h> // used by sincosf

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

/** @typedef EmlFFT
*
* FFT algorithm state
*/
typedef struct _EmlFFT {
    int length; // (n/2)
    float *sin;
    float *cos;
} EmlFFT;


EmlError
eml_fft_forward(EmlFFT table, float real[], float imag[], size_t n) {

    // Compute levels = floor(log2(n))
	int levels = 0;
	for (size_t temp = n; temp > 1U; temp >>= 1)
		levels++;

    EML_PRECONDITION(((size_t)(1U << levels)) == n, EmlSizeMismatch);
    EML_PRECONDITION((size_t)table.length == n/2, EmlSizeMismatch);

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
				float tpre =  real[l] * table.cos[k] + imag[l] * table.sin[k];
				float tpim = -real[l] * table.sin[k] + imag[l] * table.cos[k];
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
typedef struct _mp_obj_fft_t {
    mp_obj_base_t base;
    EmlFFT fft;
} mp_obj_fft_t;

mp_obj_full_type_t fft_type;

// Create a new instance
static mp_obj_t fft_new(mp_obj_t length_obj) {

    const int fft_length = mp_obj_get_int(length_obj);

    // Construct object
    mp_obj_fft_t *o = mp_obj_malloc(mp_obj_fft_t, (mp_obj_type_t *)&fft_type);
    EmlFFT *self = &o->fft;

    const int table_length = fft_length / 2;
    self->cos = (float *)m_malloc(sizeof(float)*table_length);
    self->sin = (float *)m_malloc(sizeof(float)*table_length);

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
static MP_DEFINE_CONST_FUN_OBJ_1(fft_new_obj, fft_new);

// Delete the instance
static mp_obj_t fft_del(mp_obj_t self_obj) {

    mp_obj_fft_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlFFT *self = &o->fft;

    // free allocated data
    m_free(self->cos);
    m_free(self->sin);

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

    if (array_length != length) {
        mp_raise_ValueError(MP_ERROR_TEXT("wrong array length"));
        return NULL;
    }

    return values;
}

// Fill the FFT factors
static mp_obj_t fft_fill(mp_obj_t self_obj, mp_obj_t sin_obj, mp_obj_t cos_obj) {

    mp_obj_fft_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlFFT *self = &o->fft;
    const int length = self->length;

    float *sin_values = check_extract_array(sin_obj, length);
    float *cos_values = check_extract_array(cos_obj, length);

    memcpy(self->sin, sin_values, sizeof(float)*length);
    memcpy(self->cos, cos_values, sizeof(float)*length);

    return mp_const_none;
 }
static MP_DEFINE_CONST_FUN_OBJ_3(fft_fill_obj, fft_fill);

// Compute the FFT
static mp_obj_t fft_run(mp_obj_t self_obj, mp_obj_t real_obj, mp_obj_t imag_obj) {

    mp_obj_fft_t *o = MP_OBJ_TO_PTR(self_obj);
    EmlFFT *self = &o->fft;
    const int fft_length = self->length*2;

    float *real_values = check_extract_array(real_obj, fft_length);
    float *imag_values = check_extract_array(imag_obj, fft_length);

    const EmlError err = eml_fft_forward(*self, real_values, imag_values, fft_length);
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

    mp_store_global(MP_QSTR_new, MP_OBJ_FROM_PTR(&fft_new_obj));

    fft_type.base.type = (void*)&mp_fun_table.type_type;
    fft_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    fft_type.name = MP_QSTR_emlfft;
    // methods
    mod_locals_dit_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_run), MP_OBJ_FROM_PTR(&fft_run_obj) };
    mod_locals_dit_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR___del__), MP_OBJ_FROM_PTR(&fft_del_obj) };
    mod_locals_dit_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_fill), MP_OBJ_FROM_PTR(&fft_fill_obj) };

    MP_OBJ_TYPE_SET_SLOT(&fft_type, locals_dict, (void*)&mod_locals_dit, 2);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

