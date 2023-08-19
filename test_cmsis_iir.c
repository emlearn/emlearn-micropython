
#include "arm_biquad_cascade_df1_init_q15.c"
#include "arm_biquad_cascade_df1_q15.c"

/* Compiling

    gcc -o cmsis_iir test_cmsis_iir.c -D__GNUC_PYTHON__ -I CMSIS-DSP/Source/FilteringFunctions/ -I CMSIS-DSP/Include/

*/
int main(int argc, char *argv[]) {

    arm_biquad_casd_df1_inst_q15 filter;
    #define N_STAGES 2
    q15_t coeffs[N_STAGES*6];
    q15_t state[N_STAGES*4];

    arm_biquad_cascade_df1_init_q15(&filter, N_STAGES, coeffs, state, 0); 	

    #define N_SAMPLES 100
    q15_t input[N_SAMPLES];
    q15_t out[N_SAMPLES];

    arm_biquad_cascade_df1_q15(&filter, input, out, N_SAMPLES);

    return 0;
}
